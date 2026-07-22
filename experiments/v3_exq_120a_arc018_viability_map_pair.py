#!/opt/local/bin/python3
"""
V3-EXQ-120a -- ARC-018 Viability Map Discriminative Pair (supersedes V3-EXQ-120).

Claim: ARC-018 -- "Hippocampus generates explicit rollouts and post-commitment
viability mapping."  Proposal: EXP-0017 / EVB-0014 (inherited from EXQ-120).
Supersedes: V3-EXQ-120 (run_id v3_exq_120_arc018_viability_map_pair_*).

WHY EXQ-120 WAS INVALIDATED (established 2026-07-22, reproduced empirically)
---------------------------------------------------------------------------
EXQ-120 recovered its executed action from

    argmax(agent.hippocampal.action_object_decoder(
        traj.get_action_object_sequence()[:, 0, :]))

and fed it straight to `env.step` (v3_exq_120_arc018_viability_map_pair.py:260,
363). On this substrate that round trip is a CONSTANT FUNCTION -- it pins to
class 3 regardless of the candidate -- because E2's step-0 action-object
embedding is near action-invariant. See
`HippocampalModule.candidate_first_action_class` and ree-v3/CLAUDE.md
"Action-object round trip is NOT an action source".

So VIABILITY_MAP_ON executed action 3 at every eval step while
VIABILITY_MAP_ABLATED acted uniformly at random. Action 3 walks into a wall in
`CausalGridWorldV2`, the episode never terminates, and the per-step harm-rate
DENOMINATOR inflates ~14x. Reproduced with a ZERO AGENT in
CausalGridWorldV2(size=6, num_hazards=4, use_proxy_fields=True), 50 eps x 200
steps:

    constant a=3  ->  10000/10000 steps, harm_rate 0.0059  (never terminates)
    random        ->    472/10000 steps, harm_rate 0.6886

Published EXQ-120: MAP_ON 0.0058 (20000/20000 steps) vs MAP_ABLATED 0.7471
(1412 steps). The two rows are the same numbers with no agent in the loop. The
"99.2% harm reduction" is a denominator artifact of immobility; the viability
map contributed nothing.

ARC-018 leg A retains two clean supports that do NOT use the round trip
(EXQ-042 and EXQ-053, both selecting via `agent.select_action`), so only the
EXQ-120 leg falls. This run replaces it.

WHAT THIS RE-RUN CHANGES (four things, each closing one leg of the artifact)
---------------------------------------------------------------------------
1. SELECTION goes through `agent.select_action(candidates, ticks)` -- E3's
   J(zeta), which returns the action DIRECTLY and never consults the decoder.
   This is the pattern EXQ-042 and EXQ-053 already use correctly.
2. Where a candidate's real first action is needed for REPORTING, it is read
   with `HippocampalModule.candidate_first_action_class(traj)`. The decoder
   round trip appears nowhere in this file, in either role.
3. A THIRD ARM, A2_STATIC (always action 4 = the env's (0,0) no-op), makes the
   immobility artifact structurally UNABLE to produce a PASS. C1 requires
   MAP_ON to beat the BETTER of the two trivial controls, so "stand still" is
   now a competitor rather than an unexamined confound. A2_STATIC is very
   nearly a re-run of what EXQ-120's MAP_ON arm actually did, so its numbers
   make the old result directly interpretable.
4. The PRIMARY DV is HARM PER EPISODE at a matched episode budget, not harm per
   step. Harm-per-step is exactly the statistic the immobility artifact gamed,
   so it is recorded as a diagnostic and is BARRED FROM EVERY CRITERION.
   Episode-termination counts, mean episode length, distinct cells visited per
   episode, distinct executed action classes and the cross-arm step-denominator
   ratio are all recorded per arm alongside the DVs.

WHY A STATIC CONTROL ARM AND NOT A MOBILITY GATE (a deliberate design choice)
----------------------------------------------------------------------------
Action 4 in `CausalGridWorldV2` is (0,0) -- STAY -- and E3's J(zeta) minimises
predicted harm, so "stand still" is a genuinely attractive policy, not a bug. A
locomotion gate would therefore misroute a REAL behavioural result ("viability
mapping converges on immobility") as an instrument failure -- the same category
error EXQ-120 made in the other direction. The static control resolves it
without a gate: if MAP_ON converges on stillness it cannot beat A2_STATIC and
FAILS C1 as a MEASUREMENT; if it beats both controls the advantage cannot be a
denominator artifact, because A2_STATIC has the maximal denominator by
construction. Either way the run yields information.

WHAT THE ABLATION REMOVES (unchanged from EXQ-120)
--------------------------------------------------
A1 disables BOTH components of ARC-018:
  (a) viability map construction -- residue accumulation off, terrain never
      forms;
  (b) dynamic precision update -- `E3.post_action_update()` never called.
It is a stronger ablation than merely disabling proposals: it removes the
information substrate entirely. Training keeps `nav_bias` (40% forced hazard
approach) in A0 and A1 so both accumulate comparable harm EXPOSURE and the map
has signal to form from. A2_STATIC executes a constant action and so takes no
nav_bias; its residue field is not a scored quantity -- it is a behavioural
denominator control only.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
----------------------------------------------------
DV = harm events per eval episode, a function of the realised state trajectory
and hence of the EXECUTED action sequence, produced by E3's ranking of J(zeta)
across the hippocampal candidate set. Symmetry group of that DV: (i) permutation
of candidate index order, (ii) a uniform additive constant across all candidate
scores, (iii) any monotone rescaling of all candidate scores -- none can move
the selection.
  A0_VIABILITY_MAP_ON   treatment arm; the manipulation IS the presence of a
                        populated viability map + dynamic precision feeding
                        E3 ranking. Not a relabelling, not a broadcast constant,
                        not a monotone map: not invariant.
  A1_MAP_ABLATED        replaces the whole selection with a uniform random draw
                        AND removes the map. Not invariant.
  A2_STATIC             replaces selection with a fixed no-op action, producing
                        the maximal-denominator, zero-locomotion trajectory.
                        Not invariant; that IS its job.
P1 asserts the CROSS-CANDIDATE first-action class COUNT -- the same statistic
E3's selection routes on, per the V3-EXQ-643 same-statistic rule.

NON-DEGENERACY PRECONDITION (breach -> substrate_not_ready_requeue, NOT a verdict)
---------------------------------------------------------------------------------
  P1 candidate_action_diversity  A0 only: distinct TRUE first-action classes
                                 across the candidate set (read via
                                 candidate_first_action_class, NOT the
                                 non-invertible decoder round trip), measured on
                                 a post-training positive control. If every
                                 candidate proposes the same action, E3 has
                                 nothing to choose between and the arm is a
                                 no-op regardless of how selection is routed --
                                 the EXQ-120 failure mode surviving the fix.
P1 is scoped OUT of A1 and A2: neither proposes candidates at all, so asserting
it there is structurally unsatisfiable (disposition (a) -- scope the
PRECONDITION out, the arm stays scorable). Per-arm gates are aggregated with
experiments/_lib/precondition_gate.py so a red arm can NEVER vacate a green one
(the V3-EXQ-785 whole-run-AND defect).

Note C2 (map populates) and C5 (active RBF centers) are kept as PASS CRITERIA
rather than promoted to preconditions, exactly as in EXQ-120: an unpopulated
map is a substantive negative about ARC-018's mechanism, not an instrument
failure, and routing it to substrate_not_ready would hide it.

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
-------------------------------------------------------------------
  C1 (LOAD-BEARING) harm_per_episode_on <= 0.85 x MIN(ablated, static)
                    -- MAP_ON must beat the BETTER of the two trivial controls
  C2 residue_events_min_on >= 20        (the viability map actually populated)
  C3 direction consistent vs the binding control on ALL seeds
  C4 data quality: min harm events per cell >= 20
  C5 active_centers_min_on >= 5         (map spread across the latent space)
  C6 effect >= 0.80 SD of the cross-seed delta vs the binding control AND
                    >= an absolute floor of 0.25 harm events per episode

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.
"""

import argparse
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.hippocampal.module import HippocampalModule
from ree_core.utils.config import HippocampalConfig, REEConfig

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.manifest_core import stamp_recording_core
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

EXPERIMENT_TYPE = "v3_exq_120a_arc018_viability_map_pair"
CLAIM_IDS = ["ARC-018"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-0014"
SUPERSEDES = "v3_exq_120_arc018_viability_map_pair"

# Hold-weighted-E3-readout gate (`validate_experiments.e3_hold_weighted_readout_lint`,
# pseudo-replication defect FORM 2). TRIAGED SAFE by an exact argument, not a bound.
#
# `agent.select_action` returns the HELD action on a non-E3 tick, so `eval_executed`
# IS hold-duration-weighted -- the gate's structural finding is correct. What makes it
# harmless is what that list is consumed by: its ONLY reader is `len(set(...))` (the
# `eval_executed_action_classes` diagnostic). Set CARDINALITY is EXACTLY invariant
# under duplication -- replicating a held class can neither add nor remove a class --
# so hold weighting cannot move the statistic in either direction, at any E3 cadence.
# That is stronger than the gate's "threshold-invariant => SAFE" triage rule, and it is
# the opposite of the DISQUALIFYING case: no distribution-shape quantity (entropy,
# variance, histogram mass) is derived from it. It feeds no criterion -- C1-C6 route on
# harm_per_episode, residue population and active centers, and the readiness gate P1
# reads candidate-set diversity, not this.
#
# Construct check: the diagnostic asks whether the selection mechanism reached
# BEHAVIOUR, and behaviour is the executed stream, so the per-ENV-STEP unit is the
# correct sampling unit here; restricting to fresh selections would measure a
# different construct.
#
# Scope caution: the marker is file-wide. A future edit deriving a magnitude or
# distribution-shape statistic from a select_action() return value must re-derive
# this exemption or drop it.
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "eval_executed is consumed ONLY by len(set(...)) (the eval_executed_action_classes "
    "diagnostic); set cardinality is exactly invariant under hold-duration replication, "
    "no distribution-shape statistic is derived from it, and it feeds no criterion"
)

ARM_MAP_ON = "A0_VIABILITY_MAP_ON"
ARM_MAP_ABLATED = "A1_MAP_ABLATED"
ARM_STATIC = "A2_STATIC"
ARMS = [ARM_MAP_ON, ARM_MAP_ABLATED, ARM_STATIC]

# The env's (0,0) no-op. CausalGridWorldV2.ACTIONS = {0:(-1,0), 1:(1,0),
# 2:(0,-1), 3:(0,1), 4:(0,0)} -- so 4 is STAY, and A2_STATIC is the pure
# immobility policy the EXQ-120 MAP_ON arm effectively executed.
STATIC_ACTION_IDX = 4

# --- pre-registered thresholds -------------------------------------------
THRESH_C1_REDUCTION = 0.15           # fractional reduction in harm PER EPISODE
THRESH_C2_MIN_RESIDUE_EVENTS = 20    # harm events accumulated into the map, per seed
THRESH_C4_MIN_HARM_EVENTS = 20       # per cell
THRESH_C5_MIN_ACTIVE_CENTERS = 5     # active RBF centers after training
THRESH_C6_EFFECT_SD = 0.80           # effect in SD of the cross-seed delta
THRESH_C6_ABS_FLOOR = 0.25           # absolute floor, harm events per episode

# --- non-degeneracy floor -------------------------------------------------
FLOOR_CANDIDATE_ACTION_CLASSES = 1.5   # P1: i.e. >= 2 distinct candidate actions

# Diagnostic-only comparability bound for the per-step rate. NOT a gate: a MAP_ON
# arm that legitimately survives longer inflates this ratio, and that is the success
# signature, not the artifact. EXQ-120's own ratio was 20000/1412 ~ 14.2.
DIAG_STEP_DENOMINATOR_RATIO_MAX = 3.0

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES * elite_fraction)
# at >= 2: `HippocampalModule.propose_trajectories` refits with the std over the
# elite stack, and torch's default std() is UNBIASED, so the std of a SINGLE elite
# is NaN. It propagates through the ao_std floor and NaN-poisons every subsequent
# candidate rollout, ending in `RuntimeError: probability tensor contains ... nan`
# from torch.multinomial inside e3_selector.select (ree-v3/CLAUDE.md defect 2).
# Reachable from the substrate default: num_candidates=8 x elite_fraction=0.2 gives
# int(1.6) == 1. 32 is the substrate default and yields num_elite = 6.
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
CONTROL_PROBE_STEPS = 25  # post-training positive-control probe for P1


# =========================================================================
# precondition specs -- each declares the regimes it is meaningful for
# =========================================================================
def _selection_under_test(ctx: Dict[str, Any]) -> bool:
    """P1 applies only to the arm whose SELECTION is the mechanism under test.

    Neither control arm proposes candidates at all -- A1 draws uniformly at random
    and A2 executes a fixed no-op -- so a candidate-diversity assertion is
    structurally unsatisfiable there. Disposition (a): scope the PRECONDITION out,
    the arm stays fully scorable.
    """
    return bool(ctx["uses_hippocampal_selection"])


PRECONDITION_SPECS = [
    PreconditionSpec(
        name="candidate_action_diversity",
        description=(
            "distinct TRUE first-action classes across the candidate set, read "
            "via HippocampalModule.candidate_first_action_class (which reads "
            "trajectory.actions, NOT the non-invertible decoder round trip). If "
            "every candidate proposes the same action then E3 has nothing to "
            "choose between and the arm is a no-op however selection is routed -- "
            "the EXQ-120 failure mode surviving the selection fix"
        ),
        control=(
            "post-training positive-control probe over CEM candidate sets that "
            "genuinely differ"
        ),
        threshold=FLOOR_CANDIDATE_ACTION_CLASSES,
        direction="lower",
        applies_to=_selection_under_test,
        applies_note="only the arm that proposes candidates at all",
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    """Pre-registered per-arm context consumed by the precondition gate."""
    return [
        # viability_map: residue accumulation + E3.post_action_update precision
        {"id": ARM_MAP_ON, "uses_hippocampal_selection": True,
         "viability_map": True, "nav_bias_applies": True,
         "policy": "e3_over_hippocampal_candidates"},
        {"id": ARM_MAP_ABLATED, "uses_hippocampal_selection": False,
         "viability_map": False, "nav_bias_applies": True,
         "policy": "uniform_random"},
        {"id": ARM_STATIC, "uses_hippocampal_selection": False,
         "viability_map": False, "nav_bias_applies": False,
         "policy": "constant_no_op"},
    ]


# =========================================================================
# substrate helpers
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _hazard_approach_action(env: CausalGridWorldV2, n_actions: int,
                            rng: random.Random) -> int:
    """Action moving up the hazard-proximity gradient. Fallback: random.

    Carried unchanged from EXQ-120: this is the nav_bias mechanism that
    guarantees harm EXPOSURE during training so the viability map has signal to
    accumulate from. It is a training-phase forcing function only -- it never
    runs during eval, so it cannot touch the DV.
    """
    obs_dict = env._get_observation_dict()
    world_state = obs_dict.get("world_state", None)
    if world_state is None or not env.use_proxy_fields:
        return rng.randint(0, n_actions - 1)
    # world_state[225:250] = hazard_field_view (5x5 flattened, proxy channel)
    field_view = world_state[225:250].numpy().reshape(5, 5)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # matches ACTIONS 0..3
    vals = []
    for dr, dc in deltas:
        r, c = 2 + dr, 2 + dc
        if 0 <= r < 5 and 0 <= c < 5:
            vals.append(float(field_view[r, c]))
        else:
            vals.append(-1.0)
    return int(np.argmax(vals))


def _candidate_true_action_classes(candidates) -> int:
    """Distinct TRUE first-action classes across a candidate set.

    Reads `trajectory.actions` via the canonical accessor. The alternative --
    argmax over `action_object_decoder(get_action_object_sequence()[:, 0, :])` --
    is a CONSTANT on this substrate and is what invalidated EXQ-120; it appears
    nowhere in this file, in either the selection or the reporting role.
    """
    classes = set()
    for traj in candidates:
        cls = HippocampalModule.candidate_first_action_class(traj)
        if cls is not None:
            classes.add(int(cls))
    return len(classes)


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[int], bool]:
    """Select via the CANONICAL E3 path over hippocampal viability-mapped candidates.

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta) and returns
    the action DIRECTLY. It never consults `action_object_decoder`, so the executed
    stream is a genuine function of the candidate set -- unlike EXQ-120's round-trip
    argmax, which was a constant function and made the whole MAP_ON arm inert.

    Returns (action_tensor, n_distinct_true_candidate_action_classes, used_hippocampal).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (
                _action_to_onehot(
                    rng.randint(0, n_actions - 1), n_actions, agent.device
                ),
                None,
                False,
            )
        n_true_classes = _candidate_true_action_classes(candidates)
        action = agent.select_action(candidates, ticks)
        return action, n_true_classes, True


def _probe_candidate_action_diversity(
    agent, env, n_actions, rng, n_steps: int,
) -> Dict[str, float]:
    """P1 positive control, measured AFTER training and BEFORE the scored eval.

    Reports the MINIMUM observed candidate-class count, not the mean: P1's claim is
    that the candidate set could offer E3 a choice, and a mean hides the ticks where
    it could not. Reporting the extremum keeps `measured` recomputable against the
    quantifier the precondition actually needs.
    """
    counts: List[int] = []
    roundtrip: List[Dict[str, Any]] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, n_classes, used = _select_action(
                agent, latent, ticks, n_actions, rng
            )
        if used and n_classes is not None:
            counts.append(int(n_classes))
            diag = agent.hippocampal.get_last_propose_diagnostics()
            rt = diag.get("action_object_roundtrip_recovery")
            if isinstance(rt, dict):
                roundtrip.append(rt)
        _, _harm, done, _info, obs_dict = env.step(action)
        if done:
            break

    # No candidate set observed at all -> 0.0, which FAILS P1 closed and routes the
    # arm to substrate_not_ready rather than letting a NaN ride into a comparison
    # that silently evaluates False.
    rt_true = [float(r.get("true_unique_classes") or 0) for r in roundtrip]
    rt_round = [float(r.get("roundtrip_unique_classes") or 0) for r in roundtrip]
    return {
        "candidate_true_action_classes_min": float(min(counts)) if counts else 0.0,
        "candidate_true_action_classes_mean": (
            float(statistics.fmean(counts)) if counts else 0.0
        ),
        # Documentation of the defect this re-run exists to bypass, recorded from the
        # substrate's own live diagnostic. roundtrip_unique_classes == 1 while
        # true_unique_classes > 1 IS the inert-selection signature -- expected to be
        # present here and expected to be harmless, because nothing selects that way.
        "roundtrip_true_unique_classes_mean": (
            float(statistics.fmean(rt_true)) if rt_true else 0.0
        ),
        "roundtrip_decoded_unique_classes_mean": (
            float(statistics.fmean(rt_round)) if rt_round else 0.0
        ),
    }


# =========================================================================
# one (seed, arm) cell
# =========================================================================
def run_cell(
    arm_ctx: Dict[str, Any],
    seed: int,
    full_config: Dict[str, Any],
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict[str, Any]:
    arm_id = arm_ctx["id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # Every arm is minted reuse-ELIGIBLE with the driver script EXCLUDED from the
    # hash, so a later, different-driver consumer in the ARC-018 lineage can match
    # it (mint-as-you-go; terminality is unknowable).
    with arm_cell(
        seed,
        config_slice=full_config,
        script_path=Path(__file__),
        include_driver_script_in_hash=False,
    ) as cell:
        rng = random.Random(seed)

        env = CausalGridWorldV2(
            seed=seed,
            size=6,
            num_hazards=4,
            num_resources=3,
            hazard_harm=full_config["hazard_harm"],
            env_drift_interval=5,
            env_drift_prob=0.1,
            proximity_harm_scale=full_config["proximity_harm_scale"],
            proximity_benefit_scale=full_config["proximity_harm_scale"] * 0.6,
            proximity_approach_threshold=0.15,
            hazard_field_decay=0.5,
            use_proxy_fields=True,
        )
        n_actions = env.action_dim

        config = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            self_dim=full_config["self_dim"],
            world_dim=full_config["world_dim"],
            alpha_world=full_config["alpha_world"],
            alpha_self=full_config["alpha_self"],
            reafference_action_dim=0,  # SD-007 off -- isolate ARC-018
        )
        config.latent.unified_latent_mode = False  # SD-005 split latents
        agent = REEAgent(config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_eval = min(2, eval_episodes) if dry_run else eval_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        # ---------------- TRAIN ----------------------------------------------
        agent.train()
        harm_train = 0
        steps_train = 0
        residue_events = 0
        hippo_fallbacks = 0

        for ep in range(n_warmup):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()

                # nav_bias forces hazard approach on a fraction of training steps so
                # both mapped and ablated arms accumulate comparable harm EXPOSURE.
                forced = (
                    arm_ctx["nav_bias_applies"] and rng.random() < nav_bias
                )
                if forced:
                    action = _action_to_onehot(
                        _hazard_approach_action(env, n_actions, rng),
                        n_actions, agent.device,
                    )
                    agent._last_action = action
                elif arm_ctx["policy"] == "e3_over_hippocampal_candidates":
                    action, _n_classes, used = _select_action(
                        agent, latent, ticks, n_actions, rng
                    )
                    if not used:
                        hippo_fallbacks += 1
                elif arm_ctx["policy"] == "constant_no_op":
                    action = _action_to_onehot(
                        STATIC_ACTION_IDX, n_actions, agent.device
                    )
                    agent._last_action = action
                else:
                    action = _action_to_onehot(
                        rng.randint(0, n_actions - 1), n_actions, agent.device
                    )
                    agent._last_action = action

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1

                if float(harm_signal) < 0:
                    harm_train += 1
                    if arm_ctx["viability_map"]:
                        # (a) build the viability map
                        agent.residue_field.accumulate(
                            z_world_curr,
                            harm_magnitude=abs(float(harm_signal)),
                        )
                        residue_events += 1
                        # (b) dynamic precision from prediction-error variance
                        try:
                            agent.e3.post_action_update(
                                actual_z_world=latent.z_world,
                                harm_occurred=True,
                            )
                        except (AttributeError, RuntimeError, TypeError):
                            pass
                    # ablated / static: neither accumulate nor update precision

                loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
                if loss.requires_grad:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()
                if done:
                    break

            if (ep + 1) % 50 == 0 or ep == n_warmup - 1:
                print(
                    f"  [train] seed={seed} arm={arm_id} ep {ep+1}/{n_warmup}"
                    f" harm_events={harm_train}"
                    f" harm_rate={harm_train / max(1, steps_train):.4f}"
                    f" residue_events={residue_events}",
                    flush=True,
                )

        # ---------------- readiness probe (post-training, pre-eval) ----------
        agent.eval()
        probe = {
            "candidate_true_action_classes_min": 0.0,
            "candidate_true_action_classes_mean": 0.0,
            "roundtrip_true_unique_classes_mean": 0.0,
            "roundtrip_decoded_unique_classes_mean": 0.0,
        }
        if arm_ctx["uses_hippocampal_selection"]:
            probe = _probe_candidate_action_diversity(
                agent, env, n_actions, rng,
                min(5, CONTROL_PROBE_STEPS) if dry_run else CONTROL_PROBE_STEPS,
            )

        # C5: active RBF centers in the viability map after training.
        try:
            stats = agent.residue_field.get_statistics()
            active_centers = int(stats.get("active_centers", torch.tensor(0)).item())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            active_centers = 0

        # ---------------- EVAL (behavioural DVs, matched episode budget) ------
        # No nav_bias in any arm at eval -- the DV must read the arm's own policy.
        harm_eval = 0
        steps_eval = 0
        episodes_done = 0
        episodes_terminated = 0
        episode_lengths: List[int] = []
        cells_per_episode: List[int] = []
        eval_executed: List[int] = []

        for _ in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()
            ep_steps = 0
            ep_cells = set()
            terminated = False

            for _ in range(n_steps):
                with torch.no_grad():
                    latent = agent.sense(
                        obs_dict["body_state"], obs_dict["world_state"]
                    )
                    ticks = agent.clock.advance()

                    if arm_ctx["policy"] == "e3_over_hippocampal_candidates":
                        action, _n_classes, used = _select_action(
                            agent, latent, ticks, n_actions, rng
                        )
                        if not used:
                            hippo_fallbacks += 1
                    elif arm_ctx["policy"] == "constant_no_op":
                        action = _action_to_onehot(
                            STATIC_ACTION_IDX, n_actions, agent.device
                        )
                        agent._last_action = action
                    else:
                        action = _action_to_onehot(
                            rng.randint(0, n_actions - 1), n_actions, agent.device
                        )
                        agent._last_action = action

                eval_executed.append(int(torch.argmax(action, dim=-1).item()))
                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_eval += 1
                ep_steps += 1
                if float(harm_signal) < 0:
                    harm_eval += 1
                ep_cells.add((int(env.agent_x), int(env.agent_y)))
                if done:
                    terminated = True
                    break

            episodes_done += 1
            episodes_terminated += int(terminated)
            episode_lengths.append(ep_steps)
            cells_per_episode.append(len(ep_cells))

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            # PRIMARY DV -- matched episode budget
            "harm_per_episode": float(harm_eval / max(1, episodes_done)),
            "harm_events_eval": int(harm_eval),
            "episodes_done": int(episodes_done),
            # SECONDARY / DIAGNOSTIC -- barred from every criterion
            "harm_per_step": float(harm_eval / max(1, steps_eval)),
            "total_steps_eval": int(steps_eval),
            # Denominator instrumentation: the numbers that made EXQ-120's
            # comparison meaningless are now on the record for every arm.
            "episodes_terminated": int(episodes_terminated),
            "episode_truncation_frac": float(
                1.0 - episodes_terminated / max(1, episodes_done)
            ),
            "mean_episode_len": float(
                statistics.fmean(episode_lengths) if episode_lengths else 0.0
            ),
            "mean_cells_per_episode": float(
                statistics.fmean(cells_per_episode) if cells_per_episode else 0.0
            ),
            "eval_executed_action_classes": float(len(set(eval_executed))),
            # ARC-018: viability map construction
            "residue_events_accumulated": int(residue_events),
            "active_centers": int(active_centers),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_field_harm_events": int(
                agent.residue_field.num_harm_events.item()
            ),
            # training-phase readouts
            "harm_per_step_train": float(harm_train / max(1, steps_train)),
            "harm_events_train": int(harm_train),
            "total_steps_train": int(steps_train),
            # readiness measurements
            "candidate_true_action_classes_min": float(
                probe["candidate_true_action_classes_min"]
            ),
            "candidate_true_action_classes_mean": float(
                probe["candidate_true_action_classes_mean"]
            ),
            "roundtrip_true_unique_classes_mean": float(
                probe["roundtrip_true_unique_classes_mean"]
            ),
            "roundtrip_decoded_unique_classes_mean": float(
                probe["roundtrip_decoded_unique_classes_mean"]
            ),
            # instrument health: steps where no candidate set could be formed and
            # selection fell back to random.
            "hippo_fallback_steps": int(hippo_fallbacks),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" harm_per_episode={row['harm_per_episode']:.4f}"
        f" harm_per_step={row['harm_per_step']:.4f}"
        f" mean_ep_len={row['mean_episode_len']:.1f}"
        f" terminated={row['episodes_terminated']}/{row['episodes_done']}"
        f" cells/ep={row['mean_cells_per_episode']:.2f}"
        f" residue_events={row['residue_events_accumulated']}"
        f" active_centers={row['active_centers']}",
        flush=True,
    )
    print(f"verdict: {'PASS' if row['harm_events_eval'] > 0 else 'FAIL'}", flush=True)
    return row


# =========================================================================
# analysis
# =========================================================================
def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    hpe = {a: _mean([r["harm_per_episode"] for r in by_arm[a]]) for a in ARMS}
    hps = {a: _mean([r["harm_per_step"] for r in by_arm[a]]) for a in ARMS}
    steps = {a: sum(r["total_steps_eval"] for r in by_arm[a]) for a in ARMS}

    on = hpe[ARM_MAP_ON]
    # The BINDING control is whichever trivial policy is already safer. ARC-018
    # requires the viability map to beat the BETTER of them: beating only the
    # random walker while losing to "stand still" is not evidence for post-
    # commitment viability mapping, and it is precisely the hole EXQ-120 fell
    # through (its MAP_ON arm WAS, behaviourally, the static control).
    controls = {a: hpe[a] for a in (ARM_MAP_ABLATED, ARM_STATIC)}
    binding_arm = min(controls, key=lambda a: controls[a])
    binding = controls[binding_arm]
    reduction_frac = float((binding - on) / binding) if binding > 0 else 0.0

    per_seed_delta: List[float] = []
    for s in seeds:
        a0 = next((r for r in by_arm[ARM_MAP_ON] if r["seed"] == s), None)
        ac = next((r for r in by_arm[binding_arm] if r["seed"] == s), None)
        if a0 and ac:
            per_seed_delta.append(ac["harm_per_episode"] - a0["harm_per_episode"])

    delta_mean = _mean(per_seed_delta)
    delta_sd = (
        float(statistics.pstdev(per_seed_delta)) if len(per_seed_delta) > 1 else 0.0
    )
    effect_sd = (
        float(delta_mean / delta_sd) if delta_sd > 1e-12
        else (float("inf") if delta_mean > 0 else 0.0)
    )
    seeds_consistent = sum(1 for d in per_seed_delta if d > 0)
    min_harm_events = min((r["harm_events_eval"] for r in rows), default=0)
    residue_events_min_on = min(
        (r["residue_events_accumulated"] for r in by_arm[ARM_MAP_ON]), default=0
    )
    active_centers_min_on = min(
        (r["active_centers"] for r in by_arm[ARM_MAP_ON]), default=0
    )

    denom_hi = max(steps[a] for a in ARMS)
    denom_lo = max(1, min(steps[a] for a in ARMS))
    denom_ratio = float(denom_hi / denom_lo)

    c1 = bool(reduction_frac >= THRESH_C1_REDUCTION)
    c2 = bool(residue_events_min_on >= THRESH_C2_MIN_RESIDUE_EVENTS)
    c3 = bool(seeds_consistent >= len(seeds))
    c4 = bool(min_harm_events >= THRESH_C4_MIN_HARM_EVENTS)
    c5 = bool(active_centers_min_on >= THRESH_C5_MIN_ACTIVE_CENTERS)
    c6 = bool(
        effect_sd >= THRESH_C6_EFFECT_SD and delta_mean >= THRESH_C6_ABS_FLOOR
    )

    return {
        # PRIMARY
        "harm_per_episode_by_arm": hpe,
        "binding_control_arm": binding_arm,
        "binding_control_harm_per_episode": float(binding),
        "harm_per_episode_reduction_frac": float(reduction_frac),
        "per_seed_delta_harm_per_episode": per_seed_delta,
        "delta_mean": float(delta_mean),
        "delta_sd": float(delta_sd),
        "effect_sd": float(effect_sd),
        "seeds_consistent": int(seeds_consistent),
        "min_harm_events": int(min_harm_events),
        # ARC-018 map construction
        "residue_events_min_on": int(residue_events_min_on),
        "active_centers_min_on": int(active_centers_min_on),
        # DIAGNOSTIC ONLY -- no criterion reads these
        "harm_per_step_by_arm": hps,
        "total_steps_eval_by_arm": steps,
        "step_denominator_ratio": denom_ratio,
        "harm_per_step_comparable": bool(
            denom_ratio <= DIAG_STEP_DENOMINATOR_RATIO_MAX
        ),
        "c1_harm_per_episode_pass": c1,
        "c2_map_populated_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "c5_active_centers_pass": c5,
        "c6_effect_pass": c6,
        "all_pass": bool(c1 and c2 and c3 and c4 and c5 and c6),
    }


def build_gate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm precondition gate. A red arm NEVER vacates a green one."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    gates = []
    for ctx in arm_contexts():
        arm_rows = by_arm[ctx["id"]]
        measured: Dict[str, float] = {}
        if _selection_under_test(ctx) and arm_rows:
            # worst cell, not the mean -- a single degenerate cell must not hide
            # behind an in-band average (the recomputability rule).
            measured["candidate_action_diversity"] = min(
                r["candidate_true_action_classes_min"] for r in arm_rows
            )
        gates.append(
            evaluate_arm_gate(ctx["id"], ctx, PRECONDITION_SPECS, measured=measured)
        )
    return aggregate_arm_gates(gates)


# =========================================================================
# main
# =========================================================================
def run(
    seeds: Tuple[int, ...] = (42, 123, 456),
    warmup_episodes: int = 150,
    eval_episodes: int = 40,
    steps_per_episode: int = 200,
    nav_bias: float = 0.40,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    full_config: Dict[str, Any] = {
        "env": {
            "name": "CausalGridWorldV2",
            "size": 6,
            "num_hazards": 4,
            "num_resources": 3,
            "env_drift_interval": 5,
            "env_drift_prob": 0.1,
            "proximity_approach_threshold": 0.15,
            "hazard_field_decay": 0.5,
            "use_proxy_fields": True,
        },
        "self_dim": 32,
        "world_dim": 32,
        "lr": 1e-3,
        "alpha_world": 0.9,   # SD-008: event-responsive z_world encoding
        "alpha_self": 0.3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "nav_bias": nav_bias,
        "reafference_action_dim": 0,
        "unified_latent_mode": False,
        "num_candidates": NUM_CANDIDATES,
        "schedule": {
            "warmup_episodes": warmup_episodes,
            "eval_episodes": eval_episodes,
            "steps_per_episode": steps_per_episode,
            "control_probe_steps": CONTROL_PROBE_STEPS,
        },
        "thresholds": {
            "C1_harm_per_episode_reduction": THRESH_C1_REDUCTION,
            "C2_min_residue_events_on": THRESH_C2_MIN_RESIDUE_EVENTS,
            "C3_seed_consistency": "all",
            "C4_min_harm_events": THRESH_C4_MIN_HARM_EVENTS,
            "C5_min_active_centers_on": THRESH_C5_MIN_ACTIVE_CENTERS,
            "C6_effect_sd": THRESH_C6_EFFECT_SD,
            "C6_abs_floor_harm_per_episode": THRESH_C6_ABS_FLOOR,
            "P1_floor_candidate_action_classes": FLOOR_CANDIDATE_ACTION_CLASSES,
            "static_action_idx": STATIC_ACTION_IDX,
            "diag_step_denominator_ratio_max": DIAG_STEP_DENOMINATOR_RATIO_MAX,
        },
    }

    # Design-time satisfiability audit: refuse the run before compute is spent if
    # any pre-registered precondition is unsatisfiable from the pre-registered
    # config (the V3-EXQ-785 rule). Never resolve a finding here by lowering a
    # threshold -- that converts a detected artifact into a citable result.
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS,
        arm_contexts(),
    )

    # Same class of design-time proof for the CEM elite floor (ree-v3/CLAUDE.md
    # defect 2): a single elite makes the refit std NaN, poisons every candidate
    # rollout, and crashes torch.multinomial inside e3_selector.select.
    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN, poisons ao_std, NaN-ing every "
            f"candidate rollout and crashing torch.multinomial in e3_selector. "
            f"Raise NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    print(
        f"[V3-EXQ-120a] design-audit OK: gate satisfiable, "
        f"num_elite={_num_elite} (>= {MIN_REQUIRED_ELITES})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for ctx in arm_contexts():
            rows.append(
                run_cell(
                    arm_ctx=ctx,
                    seed=seed,
                    full_config=full_config,
                    warmup_episodes=warmup_episodes,
                    eval_episodes=eval_episodes,
                    steps_per_episode=steps_per_episode,
                    nav_bias=nav_bias,
                    dry_run=dry_run,
                )
            )

    gate = build_gate(rows)
    analysis = analyse(rows, list(seeds))

    non_degenerate = bool(gate["non_degenerate"])
    if not non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        interpretation_text = (
            "SUBSTRATE NOT READY -- not a verdict on ARC-018. The hippocampal "
            "candidate set offered E3 no choice of first action, so the mapped arm "
            "is a no-op however selection is routed. " + gate["degeneracy_reason"]
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "arc018_viability_map_harm_advantage_reproduced"
            direction = "supports"
            interpretation_text = (
                "ARC-018 SUPPORTED against BOTH trivial controls: a populated "
                "viability map (residue terrain + dynamic precision) ranked by E3 "
                "reduces harm per episode by "
                f"{analysis['harm_per_episode_reduction_frac']*100:.1f}% vs the "
                f"binding control {analysis['binding_control_arm']} "
                f"({analysis['binding_control_harm_per_episode']:.3f} -> "
                f"{analysis['harm_per_episode_by_arm'][ARM_MAP_ON]:.3f} harm events "
                f"per episode at a matched episode budget), consistent on "
                f"{analysis['seeds_consistent']}/{len(seeds)} seeds, with the map "
                f"populated (min residue events "
                f"{analysis['residue_events_min_on']}, min active centers "
                f"{analysis['active_centers_min_on']}). Unlike EXQ-120 this rests on "
                "a per-EPISODE denominator and beats a pure immobility policy, so it "
                "cannot be a denominator artifact."
            )
        else:
            label = "arc018_viability_map_harm_advantage_not_reproduced"
            direction = "weakens"
            interpretation_text = (
                "ARC-018 NOT REPRODUCED on this leg once selection is routed through "
                "E3, the DV is harm per episode at a matched episode budget, and a "
                "static no-op control is present: reduction vs the binding control "
                f"{analysis['binding_control_arm']} was "
                f"{analysis['harm_per_episode_reduction_frac']*100:.1f}% "
                f"< {THRESH_C1_REDUCTION*100:.0f}% required (C2 map populated="
                f"{analysis['c2_map_populated_pass']}, C5 active centers="
                f"{analysis['c5_active_centers_pass']}). The candidate set offered E3 "
                "a genuine choice (P1 green), so this is a measurement rather than an "
                "instrument failure. Read C2/C5 before concluding: a FAIL with the map "
                "UNPOPULATED is a different finding (the mechanism never formed) from "
                "a FAIL with the map populated (it formed and did not help). Note "
                "ARC-018 leg A retains EXQ-042 and EXQ-053, neither of which uses the "
                "round trip -- this run replaces the EXQ-120 leg only, and should not "
                "be read as retiring the claim on its own."
            )

    criteria = [
        {"name": "C1_harm_per_episode_reduction", "load_bearing": True,
         "passed": analysis["c1_harm_per_episode_pass"]},
        {"name": "C2_map_populated", "load_bearing": False,
         "passed": analysis["c2_map_populated_pass"]},
        {"name": "C3_seed_consistency", "load_bearing": False,
         "passed": analysis["c3_seed_consistency_pass"]},
        {"name": "C4_data_quality", "load_bearing": False,
         "passed": analysis["c4_data_quality_pass"]},
        {"name": "C5_active_centers", "load_bearing": False,
         "passed": analysis["c5_active_centers_pass"]},
        {"name": "C6_effect_size", "load_bearing": False,
         "passed": analysis["c6_effect_pass"]},
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": gate["degeneracy_reason"],
        "per_arm_gate": gate["per_arm_gate"],
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {
                    ARM_MAP_ON: [
                        "C1_harm_per_episode_reduction",
                        "C2_map_populated",
                        "C3_seed_consistency",
                        "C5_active_centers",
                        "C6_effect_size",
                    ],
                    ARM_MAP_ABLATED: ["C4_data_quality"],
                    ARM_STATIC: [],
                },
                gate,
            ),
        },
        "criteria": criteria,
        "analysis": analysis,
        "arm_results": rows,
        "per_seed_harm_per_episode": {
            a: [r["harm_per_episode"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "per_seed_harm_per_step": {
            a: [r["harm_per_step"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "per_seed_mean_episode_len": {
            a: [r["mean_episode_len"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "registered_thresholds": full_config["thresholds"],
        "supersession_note": (
            "Supersedes V3-EXQ-120, whose MAP_ON arm selected via "
            "argmax(action_object_decoder(get_action_object_sequence()[:,0,:])) -- a "
            "CONSTANT function on this substrate (pins to class 3). That arm executed "
            "one action class, walked into a wall, never terminated, and inflated the "
            "per-step harm-rate denominator ~14x (20000 steps vs 1412 ablated). Its "
            "99.2% harm reduction is reproducible with a ZERO AGENT and is not "
            "evidence for ARC-018. Mark the EXQ-120 manifest "
            "evidence_direction: superseded. ARC-018 leg A retains EXQ-042 and "
            "EXQ-053, which select via agent.select_action and are unaffected."
        ),
        "harm_per_step_bar_note": (
            "harm_per_step and step_denominator_ratio are recorded for auditability "
            "and are BARRED FROM EVERY CRITERION. Harm-per-step is the exact statistic "
            "the EXQ-120 immobility artifact gamed, so no verdict in this script can "
            "route on it. The denominator ratio is deliberately NOT a gate either: a "
            "MAP_ON arm that legitimately survives longer inflates it, so gating would "
            "refuse the success signature. The artifact is separated from the success "
            "by the A2_STATIC control -- an immobility policy cannot be beaten by an "
            "immobility artifact -- rather than by a locomotion gate, which would "
            "misroute a real 'viability mapping converges on stillness' result as an "
            "instrument failure (see the module docstring)."
        ),
        "static_control_note": (
            "A2_STATIC executes the env's (0,0) no-op at every step and is therefore "
            "a close behavioural re-run of what EXQ-120's MAP_ON arm actually did. "
            "Compare its harm_per_step and total_steps_eval against EXQ-120's "
            "published MAP_ON row (0.0058, 20000/20000 steps) when reading this "
            "manifest: that is the artifact, reproduced deliberately and labelled."
        ),
    }

    # stamp AFTER arm_results so substrate_hash HOISTS from the per-cell fingerprints
    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0,
    )

    print("\n[V3-EXQ-120a] Results", flush=True)
    for a in ARMS:
        print(
            f"  {a}: harm_per_episode="
            f"{analysis['harm_per_episode_by_arm'][a]:.4f}"
            f"  harm_per_step={analysis['harm_per_step_by_arm'][a]:.4f}"
            f"  steps={analysis['total_steps_eval_by_arm'][a]}",
            flush=True,
        )
    print(
        f"  binding_control={analysis['binding_control_arm']}"
        f"  reduction_frac={analysis['harm_per_episode_reduction_frac']:+.4f}"
        f"  effect_sd={analysis['effect_sd']:.3f}"
        f"  seeds_consistent={analysis['seeds_consistent']}/{len(seeds)}",
        flush=True,
    )
    print(
        f"  residue_events_min_on={analysis['residue_events_min_on']}"
        f"  active_centers_min_on={analysis['active_centers_min_on']}"
        f"  step_denominator_ratio={analysis['step_denominator_ratio']:.2f}"
        f"  harm_per_step_comparable={analysis['harm_per_step_comparable']}",
        flush=True,
    )
    print(f"  non_degenerate={non_degenerate}  outcome={outcome}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--eval-eps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--nav-bias", type=float, default=0.40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        nav_bias=args.nav_bias,
        dry_run=args.dry_run,
    )

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=list(args.seeds),
        script_path=Path(__file__),
        started_at=_t_start,
    )
    print(f"\nResult written to: {out_path}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
