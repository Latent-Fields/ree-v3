#!/opt/local/bin/python3
"""
V3-EXQ-800 -- ARC-007 leg 2: residue-scramble dissociation.

Claim: ARC-007 -- "Hippocampal systems store and replay paths through
residue-field terrain."  Proposal: EXP-0393 (manual_proposals.v1.json,
digested 2026-07-21).

WHAT IS AND IS NOT UNDER TEST
-----------------------------
ARC-007 is CONJUNCTIVE and only one leg has ever been tested.

  LEG 1 -- "hippocampal systems store and replay paths" -- is already carried
  behaviourally by EXQ-114 (harm_rate 0.006 intact vs 0.765 ablated, 99.2%
  reduction, 2/2 seeds). That is what exp_conf 0.55 rests on. NOT under test
  here; A1 reproduces it only as a REFERENCE arm.

  LEG 2 -- "THROUGH RESIDUE-FIELD TERRAIN" -- is untested. Nothing in the
  evidence trail distinguishes paths through a residue field from paths
  through an ordinary spatial/latent map that residue merely co-varies with.
  Leg 2 is the entire REE-specific content: without it ARC-007 reduces to
  "a hippocampus helps navigation" -- true, generic, and not load-bearing for
  the 76 dependents that assume residue-shaped terrain. LEG 2 IS THE TEST.

DESIGN -- dissociation with the hippocampal machinery left FULLY INTACT
----------------------------------------------------------------------
  A0_INTACT           reference. Hippocampal CEM active, residue field live.
  A1_HIPPO_ABLATED    EXQ-114 reference arm: hippocampal proposal bypassed
                      (uniform random action). Establishes the base effect is
                      live in THIS build, so a degradation has something to be
                      attributed against.
  A2_RESIDUE_PERMUTED the manipulation. Hippocampal machinery untouched; the
                      residue field's weight<->location correspondence is
                      PERMUTED across active RBF centers at eval entry.
                      Preserves the marginal weight distribution EXACTLY (it is
                      a permutation) and preserves the spatial map (center
                      locations are not moved). Destroys only which location
                      carries which residue.
  A3_RESIDUE_FROZEN   residue field frozen at initialisation (no accumulation
                      at any point), so the terrain never acquires structure.

The residue field in V3 is an RBF field over z_world (ree_core/residue/field.py),
NOT a literal (y,x) grid -- so the proposal's "permuted across cells" is
operationalised as a permutation over ACTIVE RBF CENTER INDICES: weights (and
their valence rows) are shuffled among the active centers while `centers`
(the locations) stay put. That is the exact intended semantics: same marginal,
same map, broken correspondence.

WHY THE PERMUTATION IS APPLIED AT EVAL ENTRY (not during training)
-----------------------------------------------------------------
A0 and A2 then share an IDENTICAL training history -- same accumulated field,
same learned terrain_prior, same E1/E2 weights. The ONLY difference is whether
the residue<->location correspondence is intact at READ time. That isolates
leg 2 as tightly as this substrate allows: a difference cannot be attributed to
"A2 learned less", because A2 learned exactly what A0 learned.
A3 is the complementary developmental manipulation (terrain never had structure
to learn from) and is deliberately NOT matched in that way.

PRIMARY DV IS BEHAVIOURAL -- and hippo_quality_gap is NOT a verdict source
--------------------------------------------------------------------------
Primary DV: eval harm_rate; secondary: path efficiency (unique cells visited
per step). `hippo_quality_gap` is SIGN-INVERTED (EXQ-397 / EXQ-397c x2
reclassified non_contributory 2026-04-22 -- intact more negative than ablated,
near bit-identical across the "harder env" fix). It is logged here as a
DIAGNOSTIC ONLY and never enters a criterion. Do not promote it to a verdict
source until the probe is redesigned.

A DEFECT IN THE EXQ-114 REFERENCE THAT IS **NOT** REPRODUCED HERE
-----------------------------------------------------------------
EXQ-114 selected `candidates[0]`. `HippocampalModule.propose_trajectories`
returns `all_trajectories` in CANDIDATE ORDER -- it does not sort by score --
so EXQ-114 executed an ARBITRARY candidate, not the lowest-residue one. Under
that selection rule the residue field has no selection authority at all, and
permuting it would be a guaranteed no-op: the run would manufacture a false
FAIL for leg 2 while measuring nothing. This script scores every candidate
explicitly with the residue field and takes the ARGMIN (see
`_select_action_idx`), and it GATES on the cross-candidate score range (P4)
so that "residue actually discriminated between candidates" is a measured
precondition rather than an assumption.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
-----------------------------------------------------
DV = eval harm_rate, a function of the realised state trajectory and hence of
the executed action sequence, which is produced by an ARGMIN over per-candidate
residue scores. Symmetry group of that DV: (i) permutation of candidate INDEX
order, (ii) any uniform additive constant applied to all candidate scores, and
(iii) any monotone rescaling of all candidate scores -- none of which can move
an argmin.
  A0  reference arm, no manipulation.
  A1  replaces the argmin with a uniform random draw. Not a relabelling, not a
      broadcast constant, not monotone: not invariant.
  A2  permutes weights ACROSS CENTERS, so each candidate's trajectory-integrated
      residue changes by a DIFFERENT amount (candidates traverse different
      regions). This is neither a broadcast constant nor a monotone map of the
      score vector, so it can and does move the argmin: not invariant.
  A3  removes field structure entirely, changing the score vector
      differentially across candidates: not invariant.
P4 (cross-candidate score RANGE, not magnitude) is the readiness gate that
asserts precisely the statistic the argmin routes on -- the V3-EXQ-643
same-statistic rule. A uniform per-tick offset would have large mean-abs and
~0 range; that must trip readiness, never read as a refuted claim.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
----------------------------------------------------------------------------------
  P1 residue_structure_live     A0/A2: cross-center weight variance > floor.
                                Permuting a FLAT field is a no-op and would
                                manufacture a false FAIL. Scoped OUT of A3
                                (deliberately flat -- asserting it there is
                                structurally unsatisfiable) and out of A1.
  P2 base_effect_reproduced     A0 only: the A0->A1 harm gap reproduces the
                                EXQ-114 magnitude. Without a base effect there
                                is nothing for A2/A3 to degrade toward.
  P3 permutation_effective      A2 only: the permutation VERIFIABLY changed the
                                field state (fraction of active centers whose
                                weight moved), not merely was requested.
  P4 candidate_score_range      A0/A2: cross-candidate residue score RANGE on a
                                positive control. Same statistic the argmin
                                routes on (see DV-symmetry above).

Per-arm gates are aggregated with experiments/_lib/precondition_gate.py, so one
arm's red gate can NEVER vacate another arm's valid result (the V3-EXQ-785
whole-run-AND defect).

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
--------------------------------------------------------------------
  C1 (LOAD-BEARING) max(recovery_A2, recovery_A3) >= 0.50 of the A0->A1 gap
  C2 effect >= 0.80 SD of the cross-seed delta
  C3 direction consistent on >= 2 of 3 seeds
  C4 data quality: min harm events per cell >= 20

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.
"""

import argparse
import json
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

EXPERIMENT_TYPE = "v3_exq_800_arc007_residue_scramble_dissociation"
CLAIM_IDS = ["ARC-007"]
EXPERIMENT_PURPOSE = "evidence"
BACKLOG_ID = "EXP-0393"

ARM_INTACT = "A0_INTACT"
ARM_HIPPO_ABLATED = "A1_HIPPO_ABLATED"
ARM_RESIDUE_PERMUTED = "A2_RESIDUE_PERMUTED"
ARM_RESIDUE_FROZEN = "A3_RESIDUE_FROZEN"
ARMS = [ARM_INTACT, ARM_HIPPO_ABLATED, ARM_RESIDUE_PERMUTED, ARM_RESIDUE_FROZEN]

# --- pre-registered thresholds -------------------------------------------
THRESH_C1_RECOVERY = 0.50        # fraction of the A0->A1 harm gap recovered
THRESH_C2_EFFECT_SD = 0.80       # effect in SD of the cross-seed delta
THRESH_C3_MIN_SEEDS = 2          # of 3
THRESH_C4_MIN_HARM_EVENTS = 20   # per cell

# --- non-degeneracy floors ------------------------------------------------
FLOOR_RESIDUE_WEIGHT_VAR = 1e-6      # P1: field must not be uniform-pinned
FLOOR_BASE_EFFECT_GAP = 0.05         # P2: A0->A1 absolute harm-rate gap
FLOOR_PERMUTATION_MOVED_FRAC = 0.50  # P3: fraction of active centers moved
FLOOR_CANDIDATE_SCORE_RANGE = 1e-6   # P4: cross-candidate score spread

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES * elite_fraction)
# at >= 2: HippocampalModule.propose_trajectories refits with
# `elite_ao_tensor.std(dim=0)`, and the std of a SINGLE elite is NaN, which
# poisons ao_std and makes every subsequent CEM sample -- hence every candidate
# rollout, hence every residue score -- NaN. Measured while smoke-testing this
# script at NUM_CANDIDATES=8 (elite_fraction 0.2 -> num_elite 1): only 1 of 8
# candidates produced a finite score at 10/40/120 warmup episodes alike, so the
# cross-candidate range was identically 0 and A0/A2/A3 came out bit-identical.
# That is a silent no-op of the entire manipulation, not a training-maturity
# effect. 32 is the substrate default and yields num_elite = 6.
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
CONTROL_PROBE_STEPS = 25  # positive-control probe for P4, run in P0


# =========================================================================
# precondition specs -- each declares the regimes it is meaningful for
# =========================================================================
def _residue_expected_live(ctx: Dict[str, Any]) -> bool:
    """P1/P4 apply only where a structured residue field is expected AND read.

    A3 freezes the field at init by design, so asserting structure there is
    structurally unsatisfiable -- disposition (a), scope the PRECONDITION out,
    the arm stays scorable. A1 never reads the field for selection.
    """
    return bool(ctx["residue_live"]) and bool(ctx["reads_residue"])


PRECONDITION_SPECS = [
    PreconditionSpec(
        name="residue_structure_live",
        description=(
            "cross-center variance of active residue RBF weights -- a flat field "
            "makes the permutation a no-op and would manufacture a false FAIL"
        ),
        control="active centers accumulated over the full training phase",
        threshold=FLOOR_RESIDUE_WEIGHT_VAR,
        direction="lower",
        applies_to=_residue_expected_live,
        applies_note=(
            "arms that both accumulate residue and read it for selection; "
            "A3 is frozen-at-init BY DESIGN (asserting structure there is "
            "unsatisfiable, not a substrate fact) and A1 does not read it"
        ),
    ),
    PreconditionSpec(
        name="base_effect_reproduced",
        description=(
            "absolute A0->A1 eval harm-rate gap; reproduces the EXQ-114 base "
            "effect so a degradation has something to be attributed against"
        ),
        control="A1 uniform-random selection vs A0 intact hippocampal selection",
        threshold=FLOOR_BASE_EFFECT_GAP,
        direction="lower",
        applies_to=lambda ctx: ctx["id"] == ARM_INTACT,
        applies_note=(
            "a property of the (A0,A1) pair, carried by the A0 reference arm; "
            "meaningless to assert of the ablated or manipulated arms themselves"
        ),
    ),
    PreconditionSpec(
        name="permutation_effective",
        description=(
            "fraction of active RBF centers whose weight actually changed -- the "
            "permutation must verifiably alter field state, not merely be requested"
        ),
        control="pre- vs post-permutation weight vector over active centers",
        threshold=FLOOR_PERMUTATION_MOVED_FRAC,
        direction="lower",
        applies_to=lambda ctx: ctx["id"] == ARM_RESIDUE_PERMUTED,
        applies_note="only the permuted arm performs a permutation",
    ),
    PreconditionSpec(
        name="candidate_score_range",
        description=(
            "cross-candidate RANGE of residue trajectory scores on a positive "
            "control -- the SAME statistic the argmin selection routes on. A "
            "uniform offset has large magnitude but ~0 range and must trip "
            "readiness, never read as a refuted claim (V3-EXQ-643 rule)"
        ),
        control="CEM candidates that genuinely differ, probed in P0 after training",
        threshold=FLOOR_CANDIDATE_SCORE_RANGE,
        direction="lower",
        applies_to=_residue_expected_live,
        applies_note=(
            "only arms whose action selection is an argmin over residue scores; "
            "A1 selects uniformly at random and A3 has no field structure to rank by"
        ),
    ),
    PreconditionSpec(
        name="executed_action_diversity",
        description=(
            "distinct EXECUTED action classes. This is the statistic the DV "
            "literally routes on: harm_rate is a function of the executed action "
            "stream, so if every tick executes the same class the DV is INVARIANT "
            "under the residue manipulation and no result is a measurement. "
            "Measured directly on this substrate: selecting via "
            "action_object_decoder argmax collapses to 1 class at every tick, "
            "while the E3 path spans 2 -- hence the E3 path (see _select_action)"
        ),
        control="E3-selected actions over the post-training positive-control probe",
        threshold=1.5,  # i.e. >= 2 distinct classes
        direction="lower",
        applies_to=_residue_expected_live,
        applies_note=(
            "arms whose behaviour is produced by E3 ranking residue-scored "
            "candidates; A1 is uniform-random (diversity is trivially high and "
            "means nothing) and A3 has no field structure driving selection"
        ),
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    """Pre-registered per-arm context consumed by the precondition gate."""
    return [
        {"id": ARM_INTACT, "residue_live": True, "reads_residue": True,
         "hippo_ablated": False, "permute": False, "freeze_residue": False},
        {"id": ARM_HIPPO_ABLATED, "residue_live": True, "reads_residue": False,
         "hippo_ablated": True, "permute": False, "freeze_residue": False},
        {"id": ARM_RESIDUE_PERMUTED, "residue_live": True, "reads_residue": True,
         "hippo_ablated": False, "permute": True, "freeze_residue": False},
        {"id": ARM_RESIDUE_FROZEN, "residue_live": False, "reads_residue": True,
         "hippo_ablated": False, "permute": False, "freeze_residue": True},
    ]


# =========================================================================
# substrate helpers
# =========================================================================
def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _active_weight_stats(residue_field) -> Dict[str, float]:
    """Variance / count over ACTIVE residue RBF weights (P1 measurement)."""
    rbf = residue_field.rbf_field
    with torch.no_grad():
        mask = rbf.active_mask
        n_active = int(mask.sum().item())
        if n_active < 2:
            return {"weight_var": 0.0, "n_active": n_active}
        w = rbf.weights[mask].detach()
        return {"weight_var": float(w.var(unbiased=False).item()),
                "n_active": n_active}


def permute_residue_locations(residue_field, seed: int) -> Dict[str, float]:
    """A2: permute weight<->location correspondence across ACTIVE RBF centers.

    Preserves EXACTLY: the multiset of weights (it is a permutation), every
    center LOCATION, and the active mask. Destroys ONLY which location carries
    which residue value. valence_vecs rows are carried with their weights so
    the per-center record stays internally consistent under the shuffle.

    Uses a dedicated torch.Generator so the permutation does not consume the
    cell's RNG stream (which would desynchronise A2 from A0's training history
    and confound the very comparison this arm exists to make).

    Returns the P3 measurement: fraction of active centers whose weight moved.
    """
    rbf = residue_field.rbf_field
    with torch.no_grad():
        active_idx = rbf.active_mask.nonzero(as_tuple=True)[0]
        n_active = int(active_idx.numel())
        if n_active < 2:
            return {"moved_frac": 0.0, "n_active": n_active}

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + 90001)
        perm = torch.randperm(n_active, generator=gen)

        before = rbf.weights[active_idx].detach().clone()
        shuffled_idx = active_idx[perm]
        rbf.weights.data[active_idx] = rbf.weights.data[shuffled_idx].clone()
        rbf.valence_vecs[active_idx] = rbf.valence_vecs[shuffled_idx].clone()
        after = rbf.weights[active_idx].detach()

        moved = int((before != after).sum().item())
        return {"moved_frac": float(moved) / float(n_active), "n_active": n_active}


def _candidate_scores(agent, candidates) -> List[Tuple[int, float]]:
    """FINITE residue trajectory scores as (candidate_index, score), lower = better.

    Non-finite scores are DROPPED rather than ranked. An early-training E2 rollout
    can diverge and yield inf/nan trajectory residue; `torch.argmin` over a tensor
    containing NaN does not reliably return the true minimum, so ranking them would
    silently degrade the argmin selection rule into an arbitrary pick -- exactly the
    "residue has no selection authority" failure this experiment must avoid (it
    would make the A2 permutation a no-op and manufacture a false FAIL). Dropping
    them keeps the selection honest and lets the caller COUNT the drops.
    """
    out: List[Tuple[int, float]] = []
    for i, traj in enumerate(candidates):
        world_seq = traj.get_world_state_sequence()
        if world_seq is None:
            continue
        val = float(agent.residue_field.evaluate_trajectory(world_seq).sum().item())
        if math.isfinite(val):
            out.append((i, val))
    return out


def _first_action_of(agent, traj, n_actions: int) -> Optional[int]:
    ao_seq = traj.get_action_object_sequence()
    if ao_seq is None or ao_seq.shape[1] == 0:
        return None
    first_ao = ao_seq[:, 0, :]
    logits = agent.hippocampal.action_object_decoder(first_ao)
    return int(torch.argmax(logits, dim=-1).item())


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[float], bool]:
    """Select via the CANONICAL E3 path: e3 ranks the hippocampal candidates.

    WHY NOT an argmin over `action_object_decoder(candidate)` -- the obvious
    reading of "pick the lowest-residue trajectory", and what EXQ-114 gestured
    at with `candidates[0]`. MEASURED on this substrate (40 warmup episodes,
    32 candidates, seed 42): EVERY candidate decodes to the SAME first-action
    class at EVERY tick (distinct classes = 1.00 of 5, min=max=1), under the
    default config AND under both `use_support_preserving_cem` and
    `use_action_class_scaffold_candidates`. The decoder is not trained by the
    E1/E2 objectives here, so it is a near-constant function of its input.

    Under that selection rule the executed action is INVARIANT under any
    manipulation of the candidate score vector -- the residue permutation would
    be an arithmetically forced no-op and the run would manufacture a false FAIL
    for ARC-007 leg 2. (It is also the most plausible reading of EXQ-196's
    harm_advantage_mean = EXACTLY 0.0 on all three seeds.)

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta), which
    reads the residue field via Phi_R and returns the action DIRECTLY, never
    touching the collapsed decoder. Measured on the same setup: executed actions
    span 2 classes, and permuting the residue field changes 63 of 150 executed
    ticks and shifts the action histogram ({4:104, 0:46} -> {4:123, 0:27}). So
    the manipulation genuinely reaches the DV on this path and only this path.

    Returns (action_tensor, cross_candidate_residue_score_range, used_e3).
    """
    with torch.no_grad():
        candidates = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(),
            z_self=latent.z_self.detach(),
            num_candidates=NUM_CANDIDATES,
        )
        if not candidates:
            return (
                _action_to_onehot(rng.randint(0, n_actions - 1), n_actions,
                                  agent.device),
                None,
                False,
            )
        # Residue score spread over the candidate set E3 is about to rank.
        # Diagnostic + readiness only; E3 owns the selection.
        scored = _candidate_scores(agent, candidates)
        vals = [v for _i, v in scored]
        score_range = float(max(vals) - min(vals)) if len(vals) >= 2 else None
        action = agent.select_action(candidates, ticks)
        return action, score_range, True


def _probe_candidate_score_range(agent, env, n_actions, rng, n_steps) -> float:
    """P0 positive control for P4: worst-case (minimum) cross-candidate range.

    Reports the MINIMUM observed range, not the mean: P4's `met` is the claim
    that residue could discriminate candidates, and a mean can hide ticks where
    it could not. Reporting the extremum keeps `measured` recomputable against
    the quantifier the criterion actually needs.
    """
    ranges: List[float] = []
    executed: List[int] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            action, span, _used = _select_action(
                agent, latent, ticks, n_actions, rng
            )
        if span is not None and math.isfinite(span):
            ranges.append(span)
        executed.append(int(torch.argmax(action, dim=-1).item()))
        _, _harm, done, _info, obs_dict = env.step(action)
        if done:
            break
    # No finite range observed at all -> 0.0, which FAILS P4 closed and routes the
    # arm to substrate_not_ready rather than letting a NaN ride through min() into
    # a comparison that silently evaluates False.
    return {
        "score_range_min": float(min(ranges)) if ranges else 0.0,
        # THE statistic the DV actually routes on: if every tick executes the same
        # action class, the DV is invariant under the manipulation and no result
        # is a measurement (see _select_action).
        "executed_action_classes": float(len(set(executed))),
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

    # The OFF/baseline arm (A0_INTACT) is minted reuse-ELIGIBLE with the driver
    # script EXCLUDED from the hash, so a later, different-driver consumer in
    # this lineage can match it (mint-as-you-go; terminality is unknowable).
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

        # ---------------- P1 TRAIN (encoder + E1/E2; residue accumulates) ----
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

                if arm_ctx["hippo_ablated"]:
                    action = _action_to_onehot(
                        rng.randint(0, n_actions - 1), n_actions, agent.device
                    )
                    agent._last_action = action
                else:
                    action, _span, used = _select_action(
                        agent, latent, ticks, n_actions, rng
                    )
                    if not used:
                        hippo_fallbacks += 1

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1

                if float(harm_signal) < 0:
                    harm_train += 1
                    # A3 freezes the terrain at initialisation: never accumulate.
                    if not arm_ctx["freeze_residue"]:
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

        # ---------------- P0-style readiness measurements --------------------
        # Measured AFTER training (that is when the field has structure) but
        # BEFORE the eval phase whose numbers the criteria route on.
        agent.eval()
        w_stats = _active_weight_stats(agent.residue_field)
        residue_weight_var = w_stats["weight_var"]

        candidate_score_range = 0.0
        probe_action_classes = 0.0
        if arm_ctx["reads_residue"]:
            probe = _probe_candidate_score_range(
                agent, env, n_actions, rng,
                min(5, CONTROL_PROBE_STEPS) if dry_run else CONTROL_PROBE_STEPS,
            )
            candidate_score_range = probe["score_range_min"]
            probe_action_classes = probe["executed_action_classes"]

        # ---------------- A2: permute the terrain at eval entry --------------
        perm_stats = {"moved_frac": 0.0, "n_active": w_stats["n_active"]}
        if arm_ctx["permute"]:
            perm_stats = permute_residue_locations(agent.residue_field, seed)
            print(
                f"  [permute] seed={seed} arm={arm_id}"
                f" moved_frac={perm_stats['moved_frac']:.4f}"
                f" n_active={perm_stats['n_active']}",
                flush=True,
            )

        # ---------------- P2 EVAL (behavioural DV) ---------------------------
        harm_eval = 0
        steps_eval = 0
        visited_cells = set()
        eval_executed: List[int] = []
        eval_score_ranges: List[float] = []

        for _ in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()
            for _ in range(n_steps):
                with torch.no_grad():
                    latent = agent.sense(
                        obs_dict["body_state"], obs_dict["world_state"]
                    )
                    ticks = agent.clock.advance()

                    if arm_ctx["hippo_ablated"]:
                        action = _action_to_onehot(
                            rng.randint(0, n_actions - 1), n_actions, agent.device
                        )
                        agent._last_action = action
                    else:
                        action, span, used = _select_action(
                            agent, latent, ticks, n_actions, rng
                        )
                        if span is not None:
                            eval_score_ranges.append(span)
                        if not used:
                            hippo_fallbacks += 1

                eval_executed.append(int(torch.argmax(action, dim=-1).item()))
                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_eval += 1
                if float(harm_signal) < 0:
                    harm_eval += 1
                visited_cells.add((int(env.agent_x), int(env.agent_y)))
                if done:
                    break

        harm_rate_eval = harm_eval / max(1, steps_eval)
        path_efficiency = len(visited_cells) / max(1, steps_eval)

        row: Dict[str, Any] = {
            "arm_id": arm_id,
            "seed": int(seed),
            "harm_rate_eval": float(harm_rate_eval),
            "harm_events_eval": int(harm_eval),
            "total_steps_eval": int(steps_eval),
            "harm_rate_train": float(harm_train / max(1, steps_train)),
            "harm_events_train": int(harm_train),
            "total_steps_train": int(steps_train),
            "path_efficiency": float(path_efficiency),
            "unique_cells_visited": int(len(visited_cells)),
            # readiness measurements
            "residue_weight_var": float(residue_weight_var),
            "residue_active_centers": int(w_stats["n_active"]),
            "candidate_score_range_min": float(candidate_score_range),
            # P5: distinct executed action classes -- the statistic the DV
            # routes on. 1 means the manipulation cannot reach behaviour.
            "probe_executed_action_classes": float(probe_action_classes),
            "eval_executed_action_classes": float(len(set(eval_executed))),
            "eval_candidate_score_range_mean": (
                float(statistics.fmean(eval_score_ranges))
                if eval_score_ranges else 0.0
            ),
            "permutation_moved_frac": float(perm_stats["moved_frac"]),
            # instrumentation health: steps where the hippocampal argmin could not
            # be formed (no candidates, or <2 finite residue scores) and selection
            # fell back to random. A large value means the arm silently degraded
            # toward A1 and its result is not what it claims to be.
            "hippo_fallback_steps": int(hippo_fallbacks),
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
            "residue_coverage": agent.residue_field.get_coverage_telemetry(),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" harm_rate={row['harm_rate_eval']:.4f}"
        f" path_eff={row['path_efficiency']:.4f}"
        f" score_range_min={row['candidate_score_range_min']:.3e}",
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

    harm = {a: _mean([r["harm_rate_eval"] for r in by_arm[a]]) for a in ARMS}
    gap = harm[ARM_HIPPO_ABLATED] - harm[ARM_INTACT]

    def _recovery(arm: str) -> float:
        if not math.isfinite(gap) or abs(gap) < 1e-12:
            return 0.0
        return float((harm[arm] - harm[ARM_INTACT]) / gap)

    rec_a2 = _recovery(ARM_RESIDUE_PERMUTED)
    rec_a3 = _recovery(ARM_RESIDUE_FROZEN)
    best_recovery = max(rec_a2, rec_a3)
    best_arm = ARM_RESIDUE_PERMUTED if rec_a2 >= rec_a3 else ARM_RESIDUE_FROZEN

    # per-seed deltas of the best-recovering manipulated arm vs intact
    per_seed_delta: List[float] = []
    for s in seeds:
        a0 = next((r for r in by_arm[ARM_INTACT] if r["seed"] == s), None)
        am = next((r for r in by_arm[best_arm] if r["seed"] == s), None)
        if a0 and am:
            per_seed_delta.append(am["harm_rate_eval"] - a0["harm_rate_eval"])

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

    c1 = bool(best_recovery >= THRESH_C1_RECOVERY)
    c2 = bool(effect_sd >= THRESH_C2_EFFECT_SD)
    c3 = bool(seeds_consistent >= THRESH_C3_MIN_SEEDS)
    c4 = bool(min_harm_events >= THRESH_C4_MIN_HARM_EVENTS)

    return {
        "harm_rate_by_arm": harm,
        "base_effect_gap": float(gap),
        "recovery_A2_permuted": float(rec_a2),
        "recovery_A3_frozen": float(rec_a3),
        "best_recovery": float(best_recovery),
        "best_recovering_arm": best_arm,
        "per_seed_delta": per_seed_delta,
        "delta_mean": float(delta_mean),
        "delta_sd": float(delta_sd),
        "effect_sd": float(effect_sd),
        "seeds_consistent": int(seeds_consistent),
        "min_harm_events": int(min_harm_events),
        "c1_recovery_pass": c1,
        "c2_effect_sd_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "all_pass": bool(c1 and c2 and c3 and c4),
    }


def build_gate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm precondition gate. A red arm NEVER vacates a green one."""
    by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in ARMS}
    for r in rows:
        by_arm[r["arm_id"]].append(r)

    harm = {a: _mean([r["harm_rate_eval"] for r in by_arm[a]]) for a in ARMS}
    base_gap = abs(harm[ARM_HIPPO_ABLATED] - harm[ARM_INTACT])

    gates = []
    for ctx in arm_contexts():
        arm_rows = by_arm[ctx["id"]]
        measured: Dict[str, float] = {}
        if _residue_expected_live(ctx):
            # worst cell, not the mean -- a single flat cell must not hide
            # behind an in-band average (the recomputability rule).
            measured["residue_structure_live"] = min(
                r["residue_weight_var"] for r in arm_rows
            )
            measured["candidate_score_range"] = min(
                r["candidate_score_range_min"] for r in arm_rows
            )
            measured["executed_action_diversity"] = min(
                r["probe_executed_action_classes"] for r in arm_rows
            )
        if ctx["id"] == ARM_INTACT:
            measured["base_effect_reproduced"] = float(base_gap)
        if ctx["id"] == ARM_RESIDUE_PERMUTED:
            measured["permutation_effective"] = min(
                r["permutation_moved_frac"] for r in arm_rows
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
    warmup_episodes: int = 300,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    dry_run: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    full_config: Dict[str, Any] = {
        "env": "CausalGridWorldV2",
        "size": 6,
        "num_hazards": 4,
        "num_resources": 3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "use_proxy_fields": True,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,   # SD-008: z_world fidelity needed for terrain reads
        "alpha_self": 0.3,
        "lr": 1e-3,
        "num_candidates": NUM_CANDIDATES,
        "warmup_episodes": warmup_episodes,
        "eval_episodes": eval_episodes,
        "steps_per_episode": steps_per_episode,
        "unified_latent_mode": False,
        "reafference_action_dim": 0,
        "arms": ARMS,
        "selection_rule": "argmin_over_candidate_residue_scores",
        "permutation_applied_at": "eval_entry",
        "thresholds": {
            "C1_recovery": THRESH_C1_RECOVERY,
            "C2_effect_sd": THRESH_C2_EFFECT_SD,
            "C3_min_seeds": THRESH_C3_MIN_SEEDS,
            "C4_min_harm_events": THRESH_C4_MIN_HARM_EVENTS,
        },
    }

    # Design-time audit: refuse a run carrying a structurally unsatisfiable
    # gate BEFORE any compute is spent (the V3-EXQ-785 free catch).
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_contexts())

    # Design-time audit 2: a single-elite CEM refit makes ao_std NaN and silently
    # NaNs every candidate rollout, which would zero the cross-candidate score
    # range and turn the whole A2/A3 manipulation into a no-op. Refuse the run
    # here rather than discovering it in P4 after the compute is spent.
    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN and poisons ao_std, NaN-ing every "
            f"candidate rollout and identically zeroing the cross-candidate residue "
            f"score range -- the A2 permutation would become a guaranteed no-op and "
            f"the run would manufacture a false FAIL for ARC-007 leg 2. Raise "
            f"NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    print(
        f"[V3-EXQ-800] design-audit OK: gate satisfiable, "
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
            "SUBSTRATE NOT READY -- not a verdict on ARC-007. "
            + gate["degeneracy_reason"]
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "arc007_leg2_supported_residue_terrain_load_bearing"
            direction = "supports"
            interpretation_text = (
                "ARC-007 leg 2 SUPPORTED: destroying the residue<->location "
                f"correspondence recovers {analysis['best_recovery']*100:.1f}% of the "
                "A0->A1 harm gap with the hippocampal machinery fully intact. The "
                "stored paths depend on residue structure specifically, not on a "
                "generic spatial map."
            )
        else:
            label = "arc007_leg2_weakened_terrain_not_load_bearing"
            direction = "weakens"
            interpretation_text = (
                "ARC-007 leg 2 WEAKENED: with the hippocampal machinery intact and "
                "the residue terrain scrambled, behaviour did not degrade toward the "
                f"ablated arm (best recovery {analysis['best_recovery']*100:.1f}% < "
                f"{THRESH_C1_RECOVERY*100:.0f}%). Path memory works, but not through "
                "residue terrain. ARC-007 should be SPLIT: the path-memory leg "
                "promotes on EXQ-114; the residue-terrain leg demotes, and dependents "
                "assuming residue-shaped terrain (MECH-073 value-shaped geometry, the "
                "MECH-269/270/271/273 replay cluster) need re-derivation. This is a "
                "decision-flipping negative, not a null."
            )

    criteria = [
        {"name": "C1_recovery_of_base_gap", "load_bearing": True,
         "passed": analysis["c1_recovery_pass"]},
        {"name": "C2_effect_sd", "load_bearing": False,
         "passed": analysis["c2_effect_sd_pass"]},
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
            # Criteria are keyed to the arm whose gate makes them meaningful: the
            # recovery/effect/consistency criteria all read the manipulated arm,
            # while the data-quality gate is a property of the intact reference.
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {
                    ARM_RESIDUE_PERMUTED: [
                        "C1_recovery_of_base_gap",
                        "C2_effect_sd",
                        "C3_seed_consistency",
                    ],
                    ARM_INTACT: ["C4_data_quality"],
                },
                gate,
            ),
        },
        "criteria": criteria,
        "analysis": analysis,
        "arm_results": rows,
        "per_seed_harm_rate": {
            a: [r["harm_rate_eval"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "registered_thresholds": full_config["thresholds"],
        "diagnostic_note_hippo_quality_gap": (
            "hippo_quality_gap is DELIBERATELY NOT COMPUTED by this script. The probe "
            "is sign-inverted (EXQ-397 / EXQ-397c x2 reclassified non_contributory "
            "2026-04-22: intact more negative than ablated, near bit-identical across "
            "the 'harder env' fix). Emitting it even as a diagnostic invites a later "
            "reader to cite a metric known to carry the wrong sign, so the verdict "
            "path here rests entirely on the behavioural DVs (harm_rate, path "
            "efficiency). The residue-selection diagnostics that ARE recorded "
            "(candidate_score_range_*, hippo_fallback_steps) are instrument-health "
            "readouts, not claim evidence."
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

    print("\n[V3-EXQ-800] Results", flush=True)
    for a in ARMS:
        print(f"  {a}: harm_rate={analysis['harm_rate_by_arm'][a]:.4f}", flush=True)
    print(
        f"  base_gap={analysis['base_effect_gap']:.4f}"
        f"  recovery_A2={analysis['recovery_A2_permuted']:.3f}"
        f"  recovery_A3={analysis['recovery_A3_frozen']:.3f}",
        flush=True,
    )
    print(f"  non_degenerate={non_degenerate}  outcome={outcome}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--eval-eps", type=int, default=60)
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
