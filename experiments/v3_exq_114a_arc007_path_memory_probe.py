#!/opt/local/bin/python3
"""
V3-EXQ-114a -- ARC-007 Path Memory Probe (supersedes V3-EXQ-114).

Claim: ARC-007 -- "Hippocampal systems store and replay paths through
residue-field terrain."  Proposal: EXP-0007 / EVB-0007 (inherited from EXQ-114).
Supersedes: V3-EXQ-114 (run_id v3_exq_114_arc007_path_memory_probe_*).

WHY EXQ-114 WAS INVALIDATED (established 2026-07-22, reproduced empirically)
---------------------------------------------------------------------------
EXQ-114 recovered its executed action from

    argmax(agent.hippocampal.action_object_decoder(
        traj.get_action_object_sequence()[:, 0, :]))

and fed it straight to `env.step` (v3_exq_114_arc007_path_memory_probe.py:193,
288). On this substrate that round trip is a CONSTANT FUNCTION -- it pins to
class 3 regardless of the candidate -- because E2's step-0 action-object
embedding is near action-invariant (per-dim std across candidates ~0.012 against
a 0.128 gap between the two largest per-class logit means). See
`HippocampalModule.candidate_first_action_class` and ree-v3/CLAUDE.md
"Action-object round trip is NOT an action source".

So MAP_NAV executed action 3 at every eval step while MAP_ABLATED acted
uniformly at random. Action 3 walks into a wall in `CausalGridWorldV2`, the
episode never terminates, and the per-step harm-rate DENOMINATOR inflates ~14x.
Reproduced with a ZERO AGENT in CausalGridWorldV2(size=6, num_hazards=4,
use_proxy_fields=True), 50 eps x 200 steps:

    constant a=3  ->  10000/10000 steps, harm_rate 0.0059  (never terminates)
    random        ->    472/10000 steps, harm_rate 0.6886

Published EXQ-114: MAP_NAV 0.0060 (20000/20000 steps) vs MAP_ABLATED 0.7648
(1420 steps). The two rows are the same numbers with no agent in the loop. The
"99.2% harm reduction" is a denominator artifact of immobility; the viability
map contributed nothing. ARC-007's `evidence_quality_note` in claims.yaml rests
on that PASS and it is its SOLE remaining behavioural support (EXQ-397 / EXQ-397c
were reclassified non_contributory 2026-04-22).

WHAT THIS RE-RUN CHANGES (four things, each closing one leg of the artifact)
---------------------------------------------------------------------------
1. SELECTION goes through `agent.select_action(candidates, ticks)` -- E3's
   J(zeta), which returns the action DIRECTLY and never consults the decoder.
   This is the pattern EXQ-042 and EXQ-053 already use correctly.
2. Where a candidate's real first action is needed for REPORTING, it is read
   with `HippocampalModule.candidate_first_action_class(traj)` (which reads
   `trajectory.actions`, the ground truth of what the candidate is). The decoder
   round trip appears nowhere in this file, in either role.
3. A THIRD ARM, A2_STATIC (always action 4 = the env's (0,0) no-op), makes the
   immobility artifact structurally UNABLE to produce a PASS. C1 requires
   MAP_NAV to beat the BETTER of the two trivial controls, so "stand still" is
   now a competitor rather than an unexamined confound. A2_STATIC is very
   nearly a re-run of what EXQ-114's NAV arm actually did, so its numbers make
   the old result directly interpretable.
4. The PRIMARY DV is HARM PER EPISODE at a matched episode budget, not harm per
   step. Harm-per-step is exactly the statistic the immobility artifact gamed,
   so it is recorded as a diagnostic and is BARRED FROM EVERY CRITERION. No
   criterion in this script can pass on a per-step rate. Episode-termination
   counts, mean episode length, distinct cells visited per episode, distinct
   executed action classes and the cross-arm step-denominator ratio are all
   recorded per arm alongside the DVs.

WHY A STATIC CONTROL ARM AND NOT A MOBILITY GATE (a deliberate design change)
----------------------------------------------------------------------------
The first draft of this script gated on locomotion: a NAV arm visiting < 3
distinct cells per episode routed `substrate_not_ready_requeue`. The dry run
showed why that is wrong. Action 4 in `CausalGridWorldV2` is (0,0) -- STAY --
and E3's J(zeta) minimises predicted harm, so "stand still" is a genuinely
attractive policy, not a bug. A mobility gate would therefore misroute a REAL
behavioural result ("terrain-guided navigation converges on immobility") as an
instrument failure, which is the same category error EXQ-114 made in the other
direction.

The static control resolves it without a gate. If MAP_NAV converges on
stillness it cannot beat A2_STATIC, so it FAILS C1 and the run reports -- as a
measurement -- that terrain guidance buys nothing over standing still. If it
beats both controls, the advantage is real and is not a denominator artifact,
because A2_STATIC has the maximal denominator by construction. Either way the
run yields information.

WHAT IS STILL GATED, AND WHY IT IS THE RIGHT THING TO GATE
----------------------------------------------------------
The one genuine readiness question left is upstream of behaviour: does the
candidate set offer E3 anything to choose BETWEEN? If every candidate proposes
the same first action, the NAV arm is an arithmetically forced no-op however
selection is routed -- the EXQ-114 failure mode surviving the selection fix.
That is P1 `candidate_action_diversity`, measured with
`candidate_first_action_class` on a post-training positive control. Executed
action-class count and locomotion are recorded as DIAGNOSTICS rather than
gates, for the reason above.

WHY THE DENOMINATOR RATIO IS RECORDED BUT NOT GATED ON
-----------------------------------------------------
A NAV arm that legitimately avoids hazards survives longer, so a large
step-denominator ratio is ALSO the signature of success. Gating on it would
refuse the very result the experiment exists to detect. So the ratio is emitted
as `step_denominator_ratio` + `harm_per_step_comparable` for a later reader,
and the verdict routes on the per-episode DV, which is comparable at a matched
episode budget by construction.

RELATION TO V3-EXQ-800 (ARC-007 leg 2) -- these are not duplicates
-----------------------------------------------------------------
EXQ-800 tests leg 2 ("THROUGH RESIDUE-FIELD TERRAIN") by dissociation, and its
A0/A1 pair reproduces the leg-1 contrast only as an internal REFERENCE, scored
on per-step `harm_rate` with a residue-argmin diagnostic path. This script is
the leg-1 REPLACEMENT for the invalidated EXQ-114: matched-episode harm-per-
episode as the primary DV, a static no-op control arm, a candidate-diversity
readiness gate, and selection through E3. Leg 1 is what ARC-007's exp_conf
currently rests on, so it needs its own uncontaminated measurement rather than
a by-product of a leg-2 design.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
----------------------------------------------------
DV = harm events per eval episode, a function of the realised state trajectory
and hence of the EXECUTED action sequence, which is produced by E3's ranking of
J(zeta) across the hippocampal candidate set. Symmetry group of that DV:
(i) permutation of candidate index order, (ii) a uniform additive constant
across all candidate scores, (iii) any monotone rescaling of all candidate
scores -- none can move the selection.
  A0_MAP_NAV       treatment arm; the manipulation IS the presence of
                   terrain-guided proposal + E3 ranking. Not a relabelling, not
                   a broadcast constant, not a monotone map: not invariant.
  A1_MAP_ABLATED   replaces the whole selection with a uniform random draw.
                   Not invariant.
  A2_STATIC        replaces selection with a fixed no-op action. A constant
                   policy is not a symmetry of the DV -- it produces a
                   different realised trajectory (the maximal-denominator,
                   zero-locomotion one). Not invariant; that IS its job.
P1 asserts the CROSS-CANDIDATE first-action class COUNT -- the same statistic
E3's selection routes on, per the V3-EXQ-643 same-statistic rule. A candidate
set with large score magnitude but a single proposed action offers E3 no choice
and must trip readiness, never read as a refuted claim.

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
                                 the EXQ-114 failure mode surviving the fix.
P1 is scoped OUT of A1_MAP_ABLATED and A2_STATIC: neither proposes candidates
at all, so asserting it there is structurally unsatisfiable (disposition (a) --
scope the PRECONDITION out, the arm stays scorable). Per-arm gates are
aggregated with experiments/_lib/precondition_gate.py so a red arm can NEVER
vacate a green one (the V3-EXQ-785 whole-run-AND defect).

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
-------------------------------------------------------------------
  C1 (LOAD-BEARING) harm_per_episode_nav <= 0.85 x MIN(ablated, static)
                    -- NAV must beat the BETTER of the two trivial controls
  C2 effect >= 0.80 SD of the cross-seed delta vs the BINDING control AND
                    >= an absolute floor of 0.25 harm events per episode
  C3 direction consistent vs the binding control on ALL seeds
  C4 data quality: min harm events per cell >= 20

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

EXPERIMENT_TYPE = "v3_exq_114a_arc007_path_memory_probe"
CLAIM_IDS = ["ARC-007"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EVB-0007"
SUPERSEDES = "v3_exq_114_arc007_path_memory_probe"

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
# variance, histogram mass) is derived from it. It feeds no criterion -- C1-C4 route on
# harm_per_episode, and the readiness gate P1 reads candidate-set diversity, not this.
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

ARM_MAP_NAV = "A0_MAP_NAV"
ARM_MAP_ABLATED = "A1_MAP_ABLATED"
ARM_STATIC = "A2_STATIC"
ARMS = [ARM_MAP_NAV, ARM_MAP_ABLATED, ARM_STATIC]

# The env's (0,0) no-op. CausalGridWorldV2.ACTIONS = {0:(-1,0), 1:(1,0),
# 2:(0,-1), 3:(0,1), 4:(0,0)} -- so 4 is STAY, and A2_STATIC is the pure
# immobility policy the EXQ-114 NAV arm effectively executed.
STATIC_ACTION_IDX = 4

# --- pre-registered thresholds -------------------------------------------
THRESH_C1_REDUCTION = 0.15          # fractional reduction in harm PER EPISODE
THRESH_C2_EFFECT_SD = 0.80          # effect in SD of the cross-seed delta
THRESH_C2_ABS_FLOOR = 0.25          # absolute floor, harm events per episode
THRESH_C4_MIN_HARM_EVENTS = 20      # per cell

# --- non-degeneracy floor -------------------------------------------------
FLOOR_CANDIDATE_ACTION_CLASSES = 1.5   # P1: i.e. >= 2 distinct candidate actions

# Diagnostic-only comparability bound for the per-step rate. NOT a gate: a NAV arm
# that legitimately survives longer inflates this ratio, and that is the success
# signature, not the artifact (see the module docstring). EXQ-114's own ratio was
# 20000/1420 ~ 14.1.
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
CONTROL_PROBE_STEPS = 25  # post-training positive-control probe for P3


# =========================================================================
# precondition specs -- each declares the regimes it is meaningful for
# =========================================================================
def _selection_under_test(ctx: Dict[str, Any]) -> bool:
    """P1 applies only to the arm whose SELECTION is the mechanism under test.

    Neither control arm proposes candidates at all -- A1_MAP_ABLATED draws
    uniformly at random and A2_STATIC executes a fixed no-op -- so a
    candidate-diversity assertion is structurally unsatisfiable there.
    Disposition (a): scope the PRECONDITION out, the arm stays fully scorable.
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
            "an upstream failure mode that survives the selection fix"
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
        {"id": ARM_MAP_NAV, "uses_hippocampal_selection": True,
         "policy": "e3_over_hippocampal_candidates"},
        {"id": ARM_MAP_ABLATED, "uses_hippocampal_selection": False,
         "policy": "uniform_random"},
        {"id": ARM_STATIC, "uses_hippocampal_selection": False,
         "policy": "constant_no_op"},
    ]


# =========================================================================
# substrate helpers
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _candidate_true_action_classes(candidates) -> int:
    """Distinct TRUE first-action classes across a candidate set.

    Reads `trajectory.actions` via the canonical accessor. The alternative --
    argmax over `action_object_decoder(get_action_object_sequence()[:, 0, :])` --
    is a CONSTANT on this substrate and is what invalidated EXQ-114; it appears
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
    """Select via the CANONICAL E3 path over hippocampal terrain-guided candidates.

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta) and returns
    the action DIRECTLY. It never consults `action_object_decoder`, so the executed
    stream is a genuine function of the candidate set -- unlike EXQ-114's round-trip
    argmax, which was a constant function and made the whole MAP_NAV arm inert.

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
    """P3 positive control, measured AFTER training and BEFORE the scored eval.

    Reports the MINIMUM observed candidate-class count, not the mean: P3's claim is
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

    # No candidate set observed at all -> 0.0, which FAILS P3 closed and routes the
    # arm to substrate_not_ready rather than letting a NaN ride into a comparison
    # that silently evaluates False.
    rt_true = [
        float(r.get("true_unique_classes") or 0) for r in roundtrip
    ]
    rt_round = [
        float(r.get("roundtrip_unique_classes") or 0) for r in roundtrip
    ]
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
    dry_run: bool,
) -> Dict[str, Any]:
    arm_id = arm_ctx["id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # Both arms are minted reuse-ELIGIBLE with the driver script EXCLUDED from the
    # hash, so a later, different-driver consumer in the ARC-007 lineage can match
    # them (mint-as-you-go; terminality is unknowable).
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
            reafference_action_dim=0,  # SD-007 off -- isolate ARC-007
        )
        config.latent.unified_latent_mode = False  # SD-005 split latents
        agent = REEAgent(config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_eval = min(2, eval_episodes) if dry_run else eval_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        # ---------------- TRAIN (E1 + E2; residue accumulates in both arms) ---
        # The ablation is ROUTING ONLY (ARC-007 strict / Q-020): the residue field
        # accumulates harm identically in both arms, so what is removed is the
        # terrain-guided NAVIGATION, not the field.
        agent.train()
        harm_train = 0
        steps_train = 0
        hippo_fallbacks = 0

        for ep in range(n_warmup):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(n_steps):
                latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
                ticks = agent.clock.advance()
                z_world_curr = latent.z_world.detach()

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

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1

                if float(harm_signal) < 0:
                    harm_train += 1
                    agent.residue_field.accumulate(
                        z_world_curr,
                        harm_magnitude=abs(float(harm_signal)),
                    )

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
                    f" harm_rate={harm_train / max(1, steps_train):.4f}",
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

        # ---------------- EVAL (behavioural DVs, matched episode budget) ------
        harm_eval = 0
        steps_eval = 0
        episodes_done = 0
        episodes_terminated = 0
        episode_lengths: List[int] = []
        cells_per_episode: List[int] = []
        eval_executed: List[int] = []
        residue_scores: List[float] = []

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
                        else:
                            # Residue cost of the trajectory E3 was ranking. A
                            # diagnostic of terrain engagement only -- it enters no
                            # criterion.
                            traj = agent.e3._committed_trajectory
                            if traj is not None:
                                ws = traj.get_world_state_sequence()
                                if ws is not None:
                                    val = float(
                                        agent.residue_field
                                        .evaluate_trajectory(ws).mean().item()
                                    )
                                    if math.isfinite(val):
                                        residue_scores.append(val)
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

        harm_per_episode = harm_eval / max(1, episodes_done)
        harm_per_step = harm_eval / max(1, steps_eval)
        mean_episode_len = (
            float(statistics.fmean(episode_lengths)) if episode_lengths else 0.0
        )
        mean_cells_per_episode = (
            float(statistics.fmean(cells_per_episode)) if cells_per_episode else 0.0
        )

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            # PRIMARY DV -- matched episode budget
            "harm_per_episode": float(harm_per_episode),
            "harm_events_eval": int(harm_eval),
            "episodes_done": int(episodes_done),
            # SECONDARY / DIAGNOSTIC -- barred from every criterion (see docstring)
            "harm_per_step": float(harm_per_step),
            "total_steps_eval": int(steps_eval),
            # Denominator instrumentation: the numbers that made EXQ-114's
            # comparison meaningless are now on the record for every arm.
            "episodes_terminated": int(episodes_terminated),
            "episode_truncation_frac": float(
                1.0 - episodes_terminated / max(1, episodes_done)
            ),
            "mean_episode_len": float(mean_episode_len),
            "mean_cells_per_episode": float(mean_cells_per_episode),
            "eval_executed_action_classes": float(len(set(eval_executed))),
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
            # selection fell back to random. A large value means the NAV arm
            # silently degraded toward the ablated arm.
            "hippo_fallback_steps": int(hippo_fallbacks),
            "mean_selected_trajectory_residue": (
                float(statistics.fmean(residue_scores)) if residue_scores else 0.0
            ),
            "n_residue_samples": int(len(residue_scores)),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" harm_per_episode={row['harm_per_episode']:.4f}"
        f" harm_per_step={row['harm_per_step']:.4f}"
        f" mean_ep_len={row['mean_episode_len']:.1f}"
        f" terminated={row['episodes_terminated']}/{row['episodes_done']}"
        f" cells/ep={row['mean_cells_per_episode']:.2f}"
        f" action_classes={row['eval_executed_action_classes']:.0f}",
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

    nav = hpe[ARM_MAP_NAV]
    # The BINDING control is whichever trivial policy is already safer. ARC-007
    # requires terrain-guided navigation to beat the BETTER of them: beating only
    # the random walker while losing to "stand still" is not evidence for a
    # residue-shaped path memory, and it is precisely the hole EXQ-114 fell
    # through (its NAV arm WAS, behaviourally, the static control).
    controls = {a: hpe[a] for a in (ARM_MAP_ABLATED, ARM_STATIC)}
    binding_arm = min(controls, key=lambda a: controls[a])
    binding = controls[binding_arm]
    reduction_frac = float((binding - nav) / binding) if binding > 0 else 0.0

    # per-seed delta: binding control - nav (positive = NAV is safer)
    per_seed_delta: List[float] = []
    for s in seeds:
        a0 = next((r for r in by_arm[ARM_MAP_NAV] if r["seed"] == s), None)
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

    denom_hi = max(steps[a] for a in ARMS)
    denom_lo = max(1, min(steps[a] for a in ARMS))
    denom_ratio = float(denom_hi / denom_lo)

    c1 = bool(reduction_frac >= THRESH_C1_REDUCTION)
    c2 = bool(
        effect_sd >= THRESH_C2_EFFECT_SD and delta_mean >= THRESH_C2_ABS_FLOOR
    )
    c3 = bool(seeds_consistent >= len(seeds))
    c4 = bool(min_harm_events >= THRESH_C4_MIN_HARM_EVENTS)

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
        # DIAGNOSTIC ONLY -- no criterion reads these
        "harm_per_step_by_arm": hps,
        "total_steps_eval_by_arm": steps,
        "step_denominator_ratio": denom_ratio,
        "harm_per_step_comparable": bool(
            denom_ratio <= DIAG_STEP_DENOMINATOR_RATIO_MAX
        ),
        "c1_harm_per_episode_pass": c1,
        "c2_effect_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "all_pass": bool(c1 and c2 and c3 and c4),
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
            "C2_effect_sd": THRESH_C2_EFFECT_SD,
            "C2_abs_floor_harm_per_episode": THRESH_C2_ABS_FLOOR,
            "C3_seed_consistency": "all",
            "C4_min_harm_events": THRESH_C4_MIN_HARM_EVENTS,
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
        f"[V3-EXQ-114a] design-audit OK: gate satisfiable, "
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
            "SUBSTRATE NOT READY -- not a verdict on ARC-007. The hippocampal "
            "candidate set offered E3 no choice of first action, so the NAV arm "
            "is a no-op however selection is routed. " + gate["degeneracy_reason"]
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "arc007_path_memory_harm_advantage_reproduced"
            direction = "supports"
            interpretation_text = (
                "ARC-007 leg 1 SUPPORTED against BOTH trivial controls: "
                "terrain-guided hippocampal proposal ranked by E3 reduces harm "
                f"per episode by {analysis['harm_per_episode_reduction_frac']*100:.1f}% "
                f"vs the binding control {analysis['binding_control_arm']} "
                f"({analysis['binding_control_harm_per_episode']:.3f} -> "
                f"{analysis['harm_per_episode_by_arm'][ARM_MAP_NAV]:.3f} harm events "
                f"per episode at a matched episode budget), consistent on "
                f"{analysis['seeds_consistent']}/{len(seeds)} seeds. Unlike EXQ-114 "
                "this rests on a per-EPISODE denominator and beats a pure "
                "immobility policy, so it cannot be a denominator artifact."
            )
        else:
            label = "arc007_path_memory_harm_advantage_not_reproduced"
            direction = "weakens"
            interpretation_text = (
                "ARC-007 leg 1 NOT REPRODUCED once selection is routed through E3, "
                "the DV is harm per episode at a matched episode budget, and a "
                "static no-op control is present: reduction vs the binding control "
                f"{analysis['binding_control_arm']} was "
                f"{analysis['harm_per_episode_reduction_frac']*100:.1f}% "
                f"< {THRESH_C1_REDUCTION*100:.0f}% required. The candidate set "
                "offered E3 a genuine choice (P1 green), so this is a measurement "
                "rather than an instrument failure. EXQ-114's 99.2% figure was a "
                "denominator artifact of a constant action stream; ARC-007's sole "
                "behavioural support does not survive its removal, and the claim's "
                "exp_conf should be revised on this run rather than on EXQ-114. "
                "This is a decision-flipping negative, not a null. Read "
                f"{ARM_STATIC} before concluding anything about navigation: if it "
                "is the binding control, the honest reading is that standing still "
                "is at least as safe as terrain-guided navigation in this env."
            )

    criteria = [
        {"name": "C1_harm_per_episode_reduction", "load_bearing": True,
         "passed": analysis["c1_harm_per_episode_pass"]},
        {"name": "C2_effect_size", "load_bearing": False,
         "passed": analysis["c2_effect_pass"]},
        {"name": "C3_seed_consistency", "load_bearing": False,
         "passed": analysis["c3_seed_consistency_pass"]},
        {"name": "C4_data_quality", "load_bearing": False,
         "passed": analysis["c4_data_quality_pass"]},
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
                    ARM_MAP_NAV: [
                        "C1_harm_per_episode_reduction",
                        "C2_effect_size",
                        "C3_seed_consistency",
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
            "Supersedes V3-EXQ-114, whose MAP_NAV arm selected via "
            "argmax(action_object_decoder(get_action_object_sequence()[:,0,:])) -- a "
            "CONSTANT function on this substrate (pins to class 3). That arm executed "
            "one action class, walked into a wall, never terminated, and inflated the "
            "per-step harm-rate denominator ~14x (20000 steps vs 1420 ablated). Its "
            "99.2% harm reduction is reproducible with a ZERO AGENT and is not "
            "evidence for ARC-007. Mark the EXQ-114 manifest "
            "evidence_direction: superseded."
        ),
        "harm_per_step_bar_note": (
            "harm_per_step and step_denominator_ratio are recorded for auditability "
            "and are BARRED FROM EVERY CRITERION. Harm-per-step is the exact statistic "
            "the EXQ-114 immobility artifact gamed, so no verdict in this script can "
            "route on it. The denominator ratio is deliberately NOT a gate either: a "
            "NAV arm that legitimately survives longer inflates it, so gating would "
            "refuse the success signature. The artifact is separated from the success "
            "by the A2_STATIC control -- an immobility policy cannot be beaten by an "
            "immobility artifact -- rather than by a locomotion gate, which would "
            "misroute a real 'navigation converges on stillness' result as an "
            "instrument failure (see the module docstring)."
        ),
        "static_control_note": (
            "A2_STATIC executes the env's (0,0) no-op at every step and is therefore "
            "a close behavioural re-run of what EXQ-114's MAP_NAV arm actually did. "
            "Compare its harm_per_step and total_steps_eval against EXQ-114's "
            "published MAP_NAV row (0.0060, 20000/20000 steps) when reading this "
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

    print("\n[V3-EXQ-114a] Results", flush=True)
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
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
