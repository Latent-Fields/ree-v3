"""V3-EXQ-929: CEM wanting_weight selection-authority dose-response (DIAGNOSTIC).

EXPERIMENT_PURPOSE = "diagnostic"
claim_ids: [] -- see "WHY NO CLAIM TAG" below.

QUESTION
--------
Does the `wanting_weight * mean(VALENCE_WANTING)` term in
`HippocampalModule._score_trajectory` (ree_core/hippocampal/module.py:1573-1581)
have any CAUSAL AUTHORITY over which candidate trajectory the CEM elite
selection picks -- and if so, at what value of `wanting_weight` does it
acquire that authority?

This is the Level-C causal-necessity question for the wanting-specific
SCORING pathway. Design + full readiness data:
REE_assembly/evidence/planning/cem_wanting_weight_causal_ablation_design_2026-08-14.md

WHAT THIS IS NOT
----------------
This is NOT a retest of V3-EXQ-914/914a's ghost-branch finding, and NOT the
routed V3-EXQ-914b. 914/914a manipulated `use_mech293_ghost_probes` -- which
candidates are PROPOSED. This holds that channel OFF in every arm and
manipulates `config.hippocampal.wanting_weight` -- how candidates, however
proposed, are SCORED. Different config field (note also the unrelated
`GhostGoalBankConfig.wanting_weight`, config.py:1885, which this run does not
touch), different code path, different consumer, different question. 914b keeps
the CLOSED/OPEN ghost contrast and turns wanting_weight on in BOTH arms as an
enabling condition; here wanting_weight IS the manipulation.

WHY NO CLAIM TAG
----------------
Per the claim_ids-accuracy rule: a PASS here would not support any registered
claim. The primary DV measures whether a substrate pathway has selection
authority -- an instrument/admissibility fact, not a hypothesis about the
organism. Findings are reported into MECH-236's evidence_quality_note and into
the `modulatory-bias-selection-authority` substrate_queue entry (a second
confirmed call site) rather than scored. GFLAG-0033 carries the same finding
into the governance sweep.

WHY DIAGNOSTIC RATHER THAN EVIDENCE
-----------------------------------
A Step-2.5a readiness probe (2026-08-14, ree-cloud-5, numbers in the design note)
measured 0 argmin flips at the documented operating value wanting_weight=0.5
across every probed cell. So a behavioural ON/OFF contrast at that value is
structurally near-guaranteed null -- exactly the vacuity the 2026-08-13
failure_autopsy_V3-EXQ-914-914a charged as MEASURES FAILED. What is genuinely
unknown, and what this run measures, is WHERE along the wanting_weight axis the
pathway acquires authority at all.

TWO READINESS FINDINGS THIS DRIVER IS BUILT AROUND (both empirical, 2026-08-14)
------------------------------------------------------------------------------
(A) The V3-EXQ-914 driver family NEVER CALLS `agent.update_residue(harm_signal)`.
    Main-field RBF centers are activated ONLY by ResidueField.accumulate()
    (field.py:633 -> RBFLayer.add_residue -> active_mask[idx]=True), whose sole
    production call site is REEAgent.update_residue() (agent.py:9653, the
    harm_signal<0 branch). Without that call the field has ZERO active centers,
    so ResidueField.update_valence's _nearest_active_center returns None,
    VALENCE_WANTING stays identically 0, and the wanting term is inert at ANY
    wanting_weight. Measured: NUM_HAZARDS=2 with 48 harm events still gave 0
    active centers. This is strictly deeper than the autopsy's NUM_HAZARDS=0
    diagnosis. This driver calls update_residue every tick (the canonical
    _harness.py:357-361 post-action hook).
(B) Even with the field live, the wanting term's spread ACROSS CANDIDATES within
    a scoring tick is 2-3 orders of magnitude below the terrain term's, so it
    acts as a near-uniform offset -- the V3-EXQ-604c uniform-broadcast hazard.
    Flips first appear around w~50 and are common only at w~500.

E3 CADENCE / LATCHING (load-bearing for the denominator)
--------------------------------------------------------
`REEAgent.generate_trajectories` (agent.py:5581-5611) returns CACHED
`_committed_candidates` whenever `not ticks["e3_tick"]`, and the E3 cadence
default is `e3_steps_per_tick=10` (ree_core/heartbeat/clock.py). A per-env-step
re-score therefore counts one genuine CEM refit up to ~10 times. The 2026-08-14
probe did exactly that, so its reported tick counts are ~10x inflated
(~1102 env steps ~= ~110 genuine refits; the ZERO-flip reading is unaffected in
direction but its n is ~10x smaller than the probe stated). THIS DRIVER GATES
the counterfactual re-score on `ticks["e3_tick"]` and reports `n_latched_ticks`
alongside `n_scored_ticks`, so the denominator is auditable -- the convention
V3-EXQ-785a established for the E3 `last_*` latch, applied to the same hazard
one layer up.

DESIGN -- 4 ARMS x 5 SEEDS, dose-response ladder
------------------------------------------------
  ARM_W0     wanting_weight=0.0    THE ABLATION. Pathway fully removed. Also a
                                   structural negative control: its
                                   selection_flip_rate is 0 by construction.
  ARM_W05    wanting_weight=0.5    The documented operating value
                                   (HippocampalConfig.wanting_weight docstring:
                                   "Set ~0.3-0.5 for goal-directed navigation").
  ARM_W50    wanting_weight=50.0   Supra-operating. Probe: flips begin here.
  ARM_W500   wanting_weight=500.0  Supra-operating. Probe: flips in 5/5 seeds.
  ARM_W5000  wanting_weight=5000.0 POSITIVE CONTROL (P3). The instrument must
                                   detect authority SOMEWHERE or the run says
                                   nothing about the operating weight. Set at the
                                   top probed rung rather than 500 deliberately:
                                   the probe measured flips on an env-step
                                   denominator, but this driver scores only
                                   GENUINE E3 refits (~1/10 as many, see E3
                                   CADENCE above), so a control calibrated on the
                                   inflated denominator could fail on the honest
                                   one and self-route the whole run
                                   substrate_not_ready_requeue. w=5000 flipped in
                                   every probed cell (27-189 evaluations), which
                                   is the headroom a positive control should have.
                                   Decided at Step-4 smoke time, before any real
                                   run; it strengthens a control and does not
                                   touch the load-bearing C_AUTH criterion.

Everything else is IDENTICAL across arms. The MECH-293 ghost-probe stack is OFF
in every arm (that is what makes this a different test from 914/914b).

WIRING REQUIRED IN EVERY ARM (this is what makes the pathway live at all)
-------------------------------------------------------------------------
  1. tonic_5ht_enabled=True            -- else update_benefit_salience no-ops
                                          (V3-EXQ-916a change 1).
  2. use_proxy_fields=True + read info["benefit_exposure"], NOT obs_dict
                                       -- (V3-EXQ-916a change 3; obs_dict never
                                          carries the key).
  3. agent.update_benefit_salience(be, drive_level=...) once per tick
                                       -- (V3-EXQ-916a change 2 / _harness.py).
  4. agent.update_residue(harm_signal) once per tick after env.step
                                       -- Finding A above.
  5. z_goal_enabled + drive_weight=2.0 + forced update_z_goal(0.5, 0.9)
                                       -- inherited from V3-EXQ-914 unchanged, so
                                          z_goal formation is held constant and
                                          cannot confound the manipulation.

ENV CONFIG (probe-calibrated, pre-registered -- NOT an inherited default)
------------------------------------------------------------------------
NUM_HAZARDS=1. VALENCE_WANTING becomes nonzero only where harm has ALREADY
allocated a center AND benefit exposure is nonzero nearby, and in this env those
pull apart: NUM_HAZARDS=0 gives contact but often no harm at all (probe seed 43:
0 harm events -> 0 centers -> P1 fails); NUM_HAZARDS=2 gives plentiful harm but
zero benefit contact on 3/3 probed seeds. NUM_HAZARDS=1 is the only probed
setting where all 5 seeds produced both: 28-32 active centers,
|VALENCE_WANTING|max 0.014-0.243, ticks_wanting_nonzero_frac 0.50-0.95.
GRID_SIZE=16 is V3-EXQ-914's own calibration (at size 8 the 5x5 local field view
saturates near-constant: mean 0.90 stdev 0.02; at 16, mean 0.54 stdev 0.24).

Two probe caveats carried forward rather than buried:
  - `hazard_harm` is INERT in this configuration. 0.05 and 0.1 produced
    bit-identical numbers in all 10 probed cells including harm-event counts, so
    the harm driving accumulation comes from proximity_harm_scale / health
    depletion, not hazard contact. Do not tune that knob expecting an effect.
  - EARLY DEATH on 3/5 seeds (probe: 60/116/42 of a possible 200 steps). That is
    the untrained-agent death confound V3-EXQ-914 removed via NUM_HAZARDS=0,
    which is not available here since it is exactly what empties the field.
    steps_alive is reported per cell and the behavioural DV is per-step
    normalised.

NO GRADIENT TRAINING / NO PHASED P0-P1-P2
-----------------------------------------
Same as V3-EXQ-807 and V3-EXQ-914: E1/E2 stay at random init throughout, and no
head is trained on z_world / z_harm / any encoder output, so the phased-training
protocol does not apply (it governs downstream heads chasing moving latent
targets; there are none here). The mechanism under test operates purely
structurally on the residue field's geometry -- the field is built by the
agent's own harm/benefit contact during the run, which is the P0 the readiness
preconditions below actually gate on.

DVs
---
PRIMARY (mechanism), C_AUTH: `selection_flip_rate` -- the fraction of GENUINE
CEM refit ticks at which
    argmin_i( terrain_i - w * wanting_i )  !=  argmin_i( terrain_i )
computed by counterfactual re-score over the actual candidate pool. Defined at
every refit tick, not a per-step magnitude at the env's competence ceiling.
Reported with it: wanting_spread_mean, terrain_spread_mean, their ratio, and
ticks_wanting_nonzero_frac.

SECONDARY (behavioural), C_BEHAV, reported, NON-GATING: mean_resource_proximity
(obs["resource_field_view"].max() averaged over env steps) -- V3-EXQ-914's own
DV, already calibrated on this env at GRID_SIZE=16. contact_rate and steps_alive
are diagnostic context ONLY: V3-EXQ-914's docstring records contact_rate pinned
at 0.00-0.014 in every arm across 5 probe seeds (this env's foraging competence
ceiling), and that caution applies unchanged -- more strongly, in fact, since the
probe measured 4-26 contacts per cell here.

PRE-REGISTERED ACCEPTANCE CRITERIA (absolute constants, not run-derived)
------------------------------------------------------------------------
  P1 (gating) WANTING FIELD LIVE: in every arm, valence_wanting_abs_max > 0 and
     ticks_wanting_nonzero_frac >= 0.25, in >= 4/5 seeds. Failing this the run is
     uninformative about the pathway and must be reported as such, NOT as a null.
  P2 (gating) FORMATION MATCHED: valence_wanting_abs_max agrees between ARM_W0
     and each ON arm within P2_FORMATION_TOL over EPISODE 1 ONLY. Episode 1 only,
     deliberately: after that the arms visit different states and a formation
     difference is a CONSEQUENCE of the manipulation, not a confound. This is
     what isolates the SCORING weight from signal FORMATION.
  P3 (gating) INSTRUMENT CAPABLE: ARM_W500.selection_flip_rate > 0 in >= 3/5
     seeds. The positive control.
  P0 (structural negative control): ARM_W0.selection_flip_rate == 0 exactly.
  C_AUTH (load-bearing): ARM_W05.selection_flip_rate > 0 in >= 3/5 seeds.
     THE READINESS PROBE PREDICTS THIS FAILS. Pre-registered anyway because the
     probe ran 3-5 episodes on a ~10x-inflated denominator and this run is
     longer; a nonzero rate would be a genuine, informative surprise.
  C_BEHAV (reported, non-gating): mean_resource_proximity gaps vs ARM_W0.
     Interpreted ONLY in light of C_AUTH -- a behavioural gap in an arm whose
     flip rate is 0 indicates a leak through some other pathway and should be
     investigated, not reported as a wanting effect.

DV-SYMMETRY (Step 3.5 mandatory declaration, per arm)
------------------------------------------------------
`selection_flip_rate` is a RANK/argmin DV, so monotone-rescaling symmetry DOES
apply -- and that is the point: it is invariant under any transform preserving
candidate ordering, which is exactly why it detects the uniform-broadcast hazard
instead of being fooled by it. It is NOT invariant under the manipulation:
wanting_weight scales one additive term NON-uniformly across candidates whenever
the wanting field is non-constant over their world-state sequences, and the P1
precondition is what certifies that non-constancy per arm. Per arm:
  ARM_W0    manipulation is the REMOVAL of the term; trivially not invariant
            (it is the reference against which the others are measured).
  ARM_W05 / ARM_W50 / ARM_W500  each scales a per-candidate (not broadcast)
            quantity, so no arm's manipulation is a uniform additive constant and
            none is invisible to an argmin DV by arithmetic.
`mean_resource_proximity` is a per-step continuous MAGNITUDE, not derived from
any CEM-internal score and not a rank DV, so monotone-rescaling symmetry does not
apply -- V3-EXQ-914's own declaration carries over verbatim.

KNOWN OPEN SUBSTRATE ITEM OVERLAPPING THIS RUN'S PATH (Step 2.5c)
-----------------------------------------------------------------
`mech203-valence-pool-admissibility` (substrate_queue.json, status
proposed_probe_gated_2026-08-08, severity UNSET -> non-blocking) names
`ree_core/agent.py::update_benefit_salience`, this run's writer. Its own text
asks for "a non-degenerate benefit/harm valence pool read off a TRAINED
substrate" and states that what is owed first is a cheap admissibility PROBE,
not construction. This run is that probe for the WANTING half, on an UNTRAINED
substrate -- so it bears on that item without discharging it (the trained-
substrate half remains open).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_929_cem_wanting_weight_selection_authority.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.residue.field import VALENCE_WANTING  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_929_cem_wanting_weight_selection_authority"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []

# The readiness-anchor reachability guard exists for a hand-written predicate that
# claims a known-positive control reproduces a signature -- a predicate narrower
# than the state it anchors to is unmeetable by construction and mislabels an
# instrument gap as a substrate verdict (V3-EXQ-778d). None of this script's four
# preconditions has that shape, and one of them IS the reachability proof:
#   ablated_arm_flip_rate_is_zero  -- arithmetic identity (w=0 makes the
#       counterfactual re-score the identity map). Reachable by construction; it
#       IS the degeneracy definition.
#   instrument_can_detect_authority -- an EMPIRICAL positive-control ARM
#       (wanting_weight=500, 3 orders of magnitude above the documented range)
#       measured with THE SHIPPED predicate on live candidate pools inside this
#       very run. That is a stronger reachability demonstration than a frozen
#       reference-cell assertion, and it gates the load-bearing criterion: if it
#       fails, C_AUTH is marked degenerate and the run self-routes
#       substrate_not_ready_requeue rather than a substrate verdict.
#   wanting_field_live / formation_matched_episode1 -- direct measurements of
#       substrate state (field magnitude, cross-arm agreement), not
#       signature-reproduction predicates; nothing is narrowed by hand.
ANCHOR_REACHABILITY_EXEMPT = (
    "P0 is an arithmetic identity and P3 is an in-run empirical positive-control "
    "arm measured with the shipped predicate; P1/P2 are direct substrate-state "
    "measurements, not signature-reproduction predicates. No hand-written "
    "narrowed anchor exists in this script."
)

# --- arms: the wanting_weight dose-response ladder ------------------------
ARM_WEIGHTS: Dict[str, float] = {
    "ARM_W0": 0.0,
    "ARM_W05": 0.5,
    "ARM_W50": 50.0,
    "ARM_W500": 500.0,
    "ARM_W5000": 5000.0,
}
ARMS: List[str] = list(ARM_WEIGHTS)
BASELINE_ARM = "ARM_W0"
OPERATING_ARM = "ARM_W05"
POSITIVE_CONTROL_ARM = "ARM_W5000"

SEEDS: List[int] = [42, 43, 45, 46, 47]  # seed 44: recurring per-seed early-death
                                          # instability on this env config family
                                          # (V3-EXQ-868 / V3-EXQ-881 / V3-EXQ-914)

N_EPISODES = 10
STEPS_PER_EPISODE = 40

# --- env (probe-calibrated; see docstring "ENV CONFIG") -------------------
GRID_SIZE = 16
NUM_RESOURCES = 3
NUM_HAZARDS = 1
HAZARD_HARM = 0.1  # measured INERT in this configuration -- see docstring
DRIVE_WEIGHT = 2.0
FORCED_BENEFIT = 0.5
FORCED_DRIVE = 0.9

# --- pre-registered thresholds (absolute constants, NOT run-derived) ------
P1_WANTING_NONZERO_FRAC_FLOOR = 0.25   # ticks with a nonzero wanting term
P1_SEED_PASS_FRACTION = 4.0 / 5.0
P2_FORMATION_TOL = 0.05                # |wanting|max agreement, episode 1 only
P3_SEED_PASS_FRACTION = 3.0 / 5.0
C_AUTH_SEED_PASS_FRACTION = 3.0 / 5.0

# z_goal-stream liveness, pooled across arm x seed cells for the manifest block.
_ZG = ZGoalStreamAccumulator()

# One representative agent per ARM, retained so write_flat_manifest can record
# `enabled_default_off_flags` off each arm's own .config (the flags differ by arm
# -- wanting_weight is the manipulation). Deliberately NOT one per cell: holding
# every arm x seed agent alive would keep 20 nets + optimiser state resident until
# the last cell finishes, which is the hazard the ZGoalStreamAccumulator pattern
# exists to avoid. Four is negligible.
_ARM_AGENTS: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_resources=NUM_RESOURCES,
        num_hazards=NUM_HAZARDS,
        hazard_harm=HAZARD_HARM,
        resource_benefit=0.5,
        resource_respawn_on_consume=True,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.05,
        proximity_approach_threshold=0.15,
        use_proxy_fields=True,   # REQUIRED: else info["benefit_exposure"] is never populated
    )


def _config_slice(arm: str) -> Dict[str, Any]:
    return {
        "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                "num_hazards": NUM_HAZARDS, "hazard_harm": HAZARD_HARM,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "alpha_world": 0.9,
        "z_goal_enabled": True,
        "drive_weight": DRIVE_WEIGHT,
        "benefit_eval_enabled": True,
        "tonic_5ht_enabled": True,
        "wanting_weight": ARM_WEIGHTS[arm],
        # MECH-293 ghost stack OFF in EVERY arm -- this is what makes this test
        # distinct from V3-EXQ-914/914b.
        "use_mech293_ghost_probes": False,
        "forced_benefit": FORCED_BENEFIT,
        "forced_drive": FORCED_DRIVE,
    }


def _make_agent(env: CausalGridWorldV2, arm: str) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,

        # z_goal formation -- IDENTICAL in every arm (held constant, not tested).
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        benefit_eval_enabled=True,
        benefit_weight=1.0,

        # THE MANIPULATION. Everything else above and below is arm-invariant.
        wanting_weight=ARM_WEIGHTS[arm],

        # SD-014/MECH-203 serotonin master switch. Without it
        # update_benefit_salience() no-ops and VALENCE_WANTING stays 0 in every
        # arm -- the V3-EXQ-916a change-1 finding.
        tonic_5ht_enabled=True,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Counterfactual re-score (the primary instrument)
# ---------------------------------------------------------------------------
def _wanting_term(hip, traj) -> Optional[float]:
    """Reproduce _score_trajectory's wanting term for ONE candidate.

    Mirrors ree_core/hippocampal/module.py:1573-1581 exactly: mean of the
    VALENCE_WANTING component of evaluate_valence over the flattened
    world-state sequence. Recomputed here rather than called through
    _score_trajectory because the counterfactual needs BOTH the with- and
    without-wanting scores from the SAME candidate pool, which the substrate
    method cannot return (it applies the arm's own weight internally). Kept
    adjacent to _terrain_term so a substrate change to that block is a single
    obvious place to re-check.
    """
    if traj.world_states is None:
        return None
    ws = traj.get_world_state_sequence()
    if ws is None:
        return None
    b, h, d = ws.shape
    with torch.no_grad():
        val = hip.residue_field.evaluate_valence(ws.reshape(b * h, d))
    return float(val[..., VALENCE_WANTING].mean().item())


def _terrain_term(hip, traj) -> Optional[float]:
    """The ARC-007 STRICT terrain score for ONE candidate (module.py:1571)."""
    ws = traj.get_world_state_sequence()
    if ws is None:
        return None
    with torch.no_grad():
        return float(hip.residue_field.evaluate_trajectory(ws).sum().item())


def _argmin(values: List[float]) -> int:
    """CEM MINIMISES the score; ties resolve to the lowest index, matching
    _score_trajectory's own sort key (score, idx) at module.py:1489."""
    return min(range(len(values)), key=lambda i: (values[i], i))


# ---------------------------------------------------------------------------
# Single arm x seed cell
# ---------------------------------------------------------------------------
def _run_cell(arm: str, seed: int, dry_run: bool = False) -> Dict[str, Any]:
    n_episodes = 2 if dry_run else N_EPISODES
    steps_per = 8 if dry_run else STEPS_PER_EPISODE
    w = ARM_WEIGHTS[arm]

    # Boundary line -- resets the runner's episodes_in_run counter.
    print(f"Seed {seed} Condition {arm}", flush=True)

    with arm_cell(seed, config_slice=_config_slice(arm),
                  script_path=Path(__file__)) as cell:
        env = _make_env(seed)
        agent = _make_agent(env, arm)
        device = agent.device
        wd = agent.config.latent.world_dim
        hip = agent.hippocampal

        n_scored_ticks = 0     # GENUINE CEM refits (e3_tick fired)
        n_latched_ticks = 0    # env steps where generate_trajectories returned cache
        n_flips = 0
        n_wanting_nonzero = 0
        wanting_spreads: List[float] = []
        terrain_spreads: List[float] = []
        n_candidates_seen: List[int] = []

        wanting_abs_max = 0.0
        wanting_abs_max_ep1 = 0.0   # P2 reads EPISODE 1 ONLY (see docstring)
        n_active_centers_max = 0

        steps_alive = 0
        resource_contacts = 0
        harm_events = 0
        resource_proximity: List[float] = []
        zgoal_norms: List[float] = []

        for ep in range(n_episodes):
            _, obs = env.reset()
            agent.reset()
            info: Dict[str, Any] = {}   # pre-first-step read stays a safe default
            for t in range(steps_per):
                rfv = obs.get("resource_field_view")
                if rfv is not None:
                    resource_proximity.append(float(rfv.max().item()))

                obs_body = obs["body_state"].to(device)
                obs_world = obs["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", True)
                    else torch.zeros(1, wd, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                # --- C_AUTH instrument -------------------------------------
                # generate_trajectories returns CACHED _committed_candidates
                # whenever not ticks["e3_tick"] (agent.py:5596-5600), and the
                # cadence default is e3_steps_per_tick=10. Scoring a cached pool
                # would count one refit up to ~10 times, so gate on the tick and
                # report the latched count as the auditable denominator.
                if not ticks.get("e3_tick", False):
                    n_latched_ticks += 1
                elif candidates and len(candidates) > 1:
                    wt = [_wanting_term(hip, tr) for tr in candidates]
                    tt = [_terrain_term(hip, tr) for tr in candidates]
                    if all(x is not None for x in wt) and all(x is not None for x in tt):
                        n_scored_ticks += 1
                        n_candidates_seen.append(len(candidates))
                        wanting_spreads.append(max(wt) - min(wt))
                        terrain_spreads.append(max(tt) - min(tt))
                        if any(abs(x) > 0.0 for x in wt):
                            n_wanting_nonzero += 1
                        base_pick = _argmin(tt)
                        arm_pick = _argmin([b - w * x for b, x in zip(tt, wt)])
                        if arm_pick != base_pick:
                            n_flips += 1

                action = agent.select_action(candidates, ticks)

                # z_goal formation -- forced identically in every arm.
                agent.update_z_goal(
                    benefit_exposure=FORCED_BENEFIT, drive_level=FORCED_DRIVE
                )
                zg = agent.goal_state.z_goal if agent.goal_state is not None else None
                if zg is not None:
                    zgoal_norms.append(float(zg.detach().norm().item()))

                # WIRING 3: the canonical VALENCE_WANTING write. benefit_exposure
                # comes from info, never obs_dict (V3-EXQ-916a change 3).
                benefit_exposure = float(info.get("benefit_exposure", 0.0) or 0.0)
                agent.update_benefit_salience(
                    benefit_exposure, drive_level=FORCED_DRIVE
                )

                am = hip.residue_field.rbf_field.active_mask
                n_active_centers_max = max(n_active_centers_max, int(am.sum().item()))
                vv = hip.residue_field.rbf_field.valence_vecs[:, VALENCE_WANTING]
                vv_max = float(vv.abs().max().item())
                wanting_abs_max = max(wanting_abs_max, vv_max)
                if ep == 0:
                    wanting_abs_max_ep1 = max(wanting_abs_max_ep1, vv_max)

                _flat, harm_signal, done, info, obs = env.step(action)

                # WIRING 4: the canonical post-action residue hook. Without this
                # the field has ZERO active centers and the whole pathway is
                # inert -- Finding A.
                agent.update_residue(float(harm_signal))

                if float(harm_signal) < 0:
                    harm_events += 1
                elif float(harm_signal) > 0:
                    resource_contacts += 1
                steps_alive += 1
                if done:
                    break

            print(
                f"  [train] {arm} seed={seed} ep {ep + 1}/{n_episodes} "
                f"scored={n_scored_ticks} latched={n_latched_ticks} "
                f"flips={n_flips}",
                flush=True,
            )

        _ZG.observe(agent)
        _ARM_AGENTS.setdefault(arm, agent)

        def _mean(vals: List[float]) -> float:
            return float(statistics.fmean(vals)) if vals else 0.0

        wanting_spread_mean = _mean(wanting_spreads)
        terrain_spread_mean = _mean(terrain_spreads)
        row: Dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "wanting_weight": w,
            # --- primary DV ---
            "selection_flip_rate": (
                n_flips / n_scored_ticks if n_scored_ticks else 0.0
            ),
            "n_flips": int(n_flips),
            "n_scored_ticks": int(n_scored_ticks),
            "n_latched_ticks": int(n_latched_ticks),
            "mean_candidates_per_tick": _mean([float(c) for c in n_candidates_seen]),
            # --- authority context ---
            "wanting_spread_mean": wanting_spread_mean,
            "terrain_spread_mean": terrain_spread_mean,
            "wanting_authority_ratio": (
                wanting_spread_mean / terrain_spread_mean
                if terrain_spread_mean > 0 else 0.0
            ),
            "ticks_wanting_nonzero_frac": (
                n_wanting_nonzero / n_scored_ticks if n_scored_ticks else 0.0
            ),
            # --- field liveness (P1 / P2) ---
            "valence_wanting_abs_max": wanting_abs_max,
            "valence_wanting_abs_max_ep1": wanting_abs_max_ep1,
            "n_active_centers_max": int(n_active_centers_max),
            # --- behavioural, secondary / non-gating ---
            "mean_resource_proximity": _mean(resource_proximity),
            "contact_rate": (
                resource_contacts / steps_alive if steps_alive else 0.0
            ),
            "resource_contacts": int(resource_contacts),
            "harm_events": int(harm_events),
            "steps_alive": int(steps_alive),
            "zgoal_norm_mean": _mean(zgoal_norms),
        }
        cell.stamp(row)

    # Per-cell verdict: the cell is informative iff its wanting field was live.
    cell_ok = (
        row["valence_wanting_abs_max"] > 0.0
        and row["ticks_wanting_nonzero_frac"] >= P1_WANTING_NONZERO_FRAC_FLOOR
    )
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)
    return row


# ---------------------------------------------------------------------------
# Aggregation + pre-registered criteria
# ---------------------------------------------------------------------------
def _by_arm(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm"] == arm]


def _analyse(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds = len(SEEDS)

    # --- P1: wanting field live in every arm ---
    p1_per_arm: Dict[str, float] = {}
    for arm in ARMS:
        ok = sum(
            1 for r in _by_arm(rows, arm)
            if r["valence_wanting_abs_max"] > 0.0
            and r["ticks_wanting_nonzero_frac"] >= P1_WANTING_NONZERO_FRAC_FLOOR
        )
        p1_per_arm[arm] = ok / n_seeds if n_seeds else 0.0
    p1_worst_arm = min(p1_per_arm, key=lambda a: p1_per_arm[a])
    p1_measured = p1_per_arm[p1_worst_arm]
    p1_met = p1_measured >= P1_SEED_PASS_FRACTION

    # --- P2: formation matched (EPISODE 1 ONLY) ---
    base_ep1 = {r["seed"]: r["valence_wanting_abs_max_ep1"] for r in _by_arm(rows, BASELINE_ARM)}
    p2_worst = 0.0
    p2_worst_cell = None
    for arm in ARMS:
        if arm == BASELINE_ARM:
            continue
        for r in _by_arm(rows, arm):
            delta = abs(r["valence_wanting_abs_max_ep1"] - base_ep1.get(r["seed"], 0.0))
            if delta > p2_worst:
                p2_worst, p2_worst_cell = delta, f"{arm}/seed{r['seed']}"
    p2_met = p2_worst <= P2_FORMATION_TOL

    # --- P3: instrument capable (positive control) ---
    p3_ok = sum(1 for r in _by_arm(rows, POSITIVE_CONTROL_ARM) if r["selection_flip_rate"] > 0.0)
    p3_measured = p3_ok / n_seeds if n_seeds else 0.0
    p3_met = p3_measured >= P3_SEED_PASS_FRACTION

    # --- P0: structural negative control ---
    p0_max = max(
        [r["selection_flip_rate"] for r in _by_arm(rows, BASELINE_ARM)] or [0.0]
    )
    p0_met = p0_max == 0.0

    # --- C_AUTH: the load-bearing finding ---
    c_auth_ok = sum(1 for r in _by_arm(rows, OPERATING_ARM) if r["selection_flip_rate"] > 0.0)
    c_auth_measured = c_auth_ok / n_seeds if n_seeds else 0.0
    c_auth_pass = c_auth_measured >= C_AUTH_SEED_PASS_FRACTION

    def _arm_mean(arm: str, key: str) -> float:
        vals = [r[key] for r in _by_arm(rows, arm)]
        return float(statistics.fmean(vals)) if vals else 0.0

    per_arm = {
        arm: {
            "wanting_weight": ARM_WEIGHTS[arm],
            "selection_flip_rate_mean": _arm_mean(arm, "selection_flip_rate"),
            "n_seeds_with_any_flip": sum(
                1 for r in _by_arm(rows, arm) if r["selection_flip_rate"] > 0.0
            ),
            "wanting_authority_ratio_mean": _arm_mean(arm, "wanting_authority_ratio"),
            "ticks_wanting_nonzero_frac_mean": _arm_mean(arm, "ticks_wanting_nonzero_frac"),
            "valence_wanting_abs_max_mean": _arm_mean(arm, "valence_wanting_abs_max"),
            "n_scored_ticks_total": sum(r["n_scored_ticks"] for r in _by_arm(rows, arm)),
            "n_latched_ticks_total": sum(r["n_latched_ticks"] for r in _by_arm(rows, arm)),
            "mean_resource_proximity_mean": _arm_mean(arm, "mean_resource_proximity"),
            "contact_rate_mean": _arm_mean(arm, "contact_rate"),
            "steps_alive_mean": _arm_mean(arm, "steps_alive"),
        }
        for arm in ARMS
    }

    base_prox = per_arm[BASELINE_ARM]["mean_resource_proximity_mean"]
    c_behav = {
        arm: per_arm[arm]["mean_resource_proximity_mean"] - base_prox
        for arm in ARMS if arm != BASELINE_ARM
    }

    gates_met = p1_met and p2_met and p3_met and p0_met
    outcome = "PASS" if (gates_met and c_auth_pass) else "FAIL"

    # Non-degeneracy: a criterion is degenerate when its own precondition did not
    # hold. C_AUTH cannot discriminate unless the field was live (P1) AND the
    # instrument can detect authority at all (P3).
    criteria_non_degenerate = {
        "P0": True,                    # structural, always meaningful
        "P1": bool(any(r["n_scored_ticks"] > 0 for r in rows)),
        "P2": bool(p1_met),            # formation comparison is vacuous on a dead field
        "P3": bool(p1_met),
        "C_AUTH": bool(p1_met and p3_met),
    }

    if not p1_met:
        label = "substrate_not_ready_requeue"
    elif not p3_met:
        label = "substrate_not_ready_requeue"
    elif not p2_met:
        label = "formation_not_matched_requeue"
    elif c_auth_pass:
        label = "wanting_scoring_has_selection_authority_at_operating_weight"
    else:
        label = "wanting_scoring_lacks_selection_authority_at_operating_weight"

    preconditions = [
        {
            "name": "wanting_field_live",
            "kind": "readiness",
            "description": (
                "worst arm's fraction of seeds with a nonzero VALENCE_WANTING "
                "field and ticks_wanting_nonzero_frac >= "
                f"{P1_WANTING_NONZERO_FRAC_FLOOR}"
            ),
            "control": (
                "the residue field is built by the agent's own harm/benefit "
                "contact during the run; NUM_HAZARDS=1 is the probe-calibrated "
                "setting where all 5 seeds produced both"
            ),
            "measured": p1_measured,
            "threshold": P1_SEED_PASS_FRACTION,
            "direction": "lower",
            "offending_cell": p1_worst_arm,
            "certifies_arms": ARMS,
            "met": bool(p1_met),
        },
        {
            "name": "formation_matched_episode1",
            "kind": "readiness",
            "description": (
                "worst |valence_wanting_abs_max_ep1 - baseline| across ON arms; "
                "episode 1 only, before behavioural divergence can feed back"
            ),
            "control": "ARM_W0 (wanting term removed) on the same seed",
            "measured": p2_worst,
            "threshold": P2_FORMATION_TOL,
            "direction": "upper",
            "offending_cell": p2_worst_cell,
            "certifies_arms": [a for a in ARMS if a != BASELINE_ARM],
            "met": bool(p2_met),
        },
        {
            "name": "instrument_can_detect_authority",
            "kind": "readiness",
            "description": (
                "fraction of seeds where the positive-control arm "
                f"({POSITIVE_CONTROL_ARM}, wanting_weight="
                f"{ARM_WEIGHTS[POSITIVE_CONTROL_ARM]}) flips at least one argmin"
            ),
            "control": (
                "supra-operating wanting_weight, 3 orders of magnitude above the "
                "documented range -- if even this never flips, the DV cannot "
                "detect authority and no reading of the operating arm is warranted"
            ),
            "measured": p3_measured,
            "threshold": P3_SEED_PASS_FRACTION,
            "direction": "lower",
            "certifies_arms": ARMS,
            "met": bool(p3_met),
        },
        {
            "name": "ablated_arm_flip_rate_is_zero",
            "kind": "negative_control",
            "description": (
                "structural negative control: with wanting_weight=0 the "
                "counterfactual re-score is the identity, so any nonzero flip "
                "rate here is an instrument bug, not a finding"
            ),
            "control": "ARM_W0, arithmetically forced",
            "measured": p0_max,
            "threshold": 0.0,
            "direction": "upper",
            "certifies_arms": [BASELINE_ARM],
            "met": bool(p0_met),
        },
    ]

    return {
        "outcome": outcome,
        "per_arm": per_arm,
        "c_auth_seed_fraction": c_auth_measured,
        "c_auth_pass": bool(c_auth_pass),
        "c_behav_proximity_gap_vs_ablated": c_behav,
        "gates_met": bool(gates_met),
        "interpretation": {
            "label": label,
            "combination_rule": (
                "PASS = P0 AND P1 AND P2 AND P3 AND C_AUTH (plain AND of all "
                "four gates and the load-bearing criterion). C_BEHAV is reported "
                "and never gates."
            ),
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": [
                {"name": "P0_ablated_flip_rate_zero", "load_bearing": False, "passed": bool(p0_met)},
                {"name": "P1_wanting_field_live", "load_bearing": False, "passed": bool(p1_met)},
                {"name": "P2_formation_matched", "load_bearing": False, "passed": bool(p2_met)},
                {"name": "P3_instrument_capable", "load_bearing": False, "passed": bool(p3_met)},
                {"name": "C_AUTH_operating_weight_has_authority",
                 "load_bearing": True, "passed": bool(c_auth_pass)},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, dry_run=dry_run))

    analysis = _analyse(rows)

    manifest: Dict[str, Any] = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_"
            f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "outcome": analysis["outcome"],
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": bool(dry_run),
        "arm_results": rows,
        "per_arm": analysis["per_arm"],
        "c_auth_seed_fraction": analysis["c_auth_seed_fraction"],
        "c_auth_pass": analysis["c_auth_pass"],
        "c_behav_proximity_gap_vs_ablated": analysis["c_behav_proximity_gap_vs_ablated"],
        "c_behav_note": (
            "READ C_BEHAV ONLY VIA C_AUTH. A Step-4 engagement check (2026-08-14, "
            "full-length cells, seeds 42/43) measured ARM_W5000 flipping the CEM "
            "elite-selection argmin on 43/104 and 96/98 GENUINE refits while its "
            "mean_resource_proximity stayed BIT-IDENTICAL to ARM_W0 (0.6480 / "
            "0.6839). So on this substrate a flipped CEM argmin does not, by "
            "itself, change behaviour -- E3's own downstream action selection "
            "(agent.py:11415/11432/11449) re-scores, and the flipped candidate "
            "evidently yields the same executed action class. A null C_BEHAV is "
            "therefore the EXPECTED reading even in an arm with full selection "
            "authority, and must NOT be reported as evidence that the wanting "
            "pathway is behaviourally inert -- that would confound two separate "
            "links in the chain. A NONZERO C_BEHAV gap in an arm whose flip rate "
            "is 0 is the genuinely anomalous case and indicates a leak through "
            "some other pathway."
        ),
        "interpretation": analysis["interpretation"],
        "pre_registered_thresholds": {
            "P1_WANTING_NONZERO_FRAC_FLOOR": P1_WANTING_NONZERO_FRAC_FLOOR,
            "P1_SEED_PASS_FRACTION": P1_SEED_PASS_FRACTION,
            "P2_FORMATION_TOL": P2_FORMATION_TOL,
            "P3_SEED_PASS_FRACTION": P3_SEED_PASS_FRACTION,
            "C_AUTH_SEED_PASS_FRACTION": C_AUTH_SEED_PASS_FRACTION,
        },
        "custom_information": {
            "manipulated_field": "REEConfig.hippocampal.wanting_weight",
            "consumer_call_site": (
                "ree_core/hippocampal/module.py::HippocampalModule._score_trajectory"
            ),
            "writer_call_site": "ree_core/agent.py::REEAgent.update_benefit_salience",
            "ghost_probe_stack": "OFF in every arm (distinct from V3-EXQ-914/914b)",
            "e3_cadence_note": (
                "generate_trajectories returns cached candidates when not "
                "ticks['e3_tick'] (agent.py:5596); the counterfactual re-score is "
                "gated on the tick and n_latched_ticks is reported so the "
                "denominator is auditable"
            ),
            "open_substrate_overlap": (
                "mech203-valence-pool-admissibility (severity unset, non-blocking) "
                "names update_benefit_salience; this run is its admissibility probe "
                "for the WANTING half on an UNTRAINED substrate and does not "
                "discharge its trained-substrate half"
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    # Non-degeneracy net (applies to every purpose): if the wanting field never
    # came alive, the run measured nothing about the pathway and must not be read
    # as a null.
    if not analysis["interpretation"]["criteria_non_degenerate"]["C_AUTH"]:
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "C_AUTH cannot discriminate: the wanting field was not live in every "
            "arm (P1) and/or the positive control never flipped an argmin (P3), "
            "so a zero flip rate at the operating weight is uninformative"
        )

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config={
            "env": {"size": GRID_SIZE, "num_resources": NUM_RESOURCES,
                    "num_hazards": NUM_HAZARDS, "hazard_harm": HAZARD_HARM,
                    "use_proxy_fields": True},
            "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
            "arms": ARM_WEIGHTS,
            "drive_weight": DRIVE_WEIGHT,
            "forced_benefit": FORCED_BENEFIT,
            "forced_drive": FORCED_DRIVE,
            "tonic_5ht_enabled": True,
            "alpha_world": 0.9,
        },
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
        agent=list(_ARM_AGENTS.values()),
    )
    return {"outcome": analysis["outcome"], "manifest_path": out_path,
            "analysis": analysis}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = result["manifest_path"]
    analysis = result["analysis"]

    print("", flush=True)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label:   {analysis['interpretation']['label']}", flush=True)
    for arm in ARMS:
        pa = analysis["per_arm"][arm]
        print(
            f"  {arm:9s} w={pa['wanting_weight']:>7.1f} "
            f"flip_rate={pa['selection_flip_rate_mean']:.4f} "
            f"seeds_with_flip={pa['n_seeds_with_any_flip']}/{len(SEEDS)} "
            f"authority_ratio={pa['wanting_authority_ratio_mean']:.3g} "
            f"scored={pa['n_scored_ticks_total']} latched={pa['n_latched_ticks_total']} "
            f"prox={pa['mean_resource_proximity_mean']:.4f}",
            flush=True,
        )
    print(f"manifest: {out_path}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
