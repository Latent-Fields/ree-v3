#!/opt/local/bin/python3
"""
V3-EXQ-266b -- Q-020 Valence Geometry Discriminative Pair (supersedes V3-EXQ-266 / 266a).

Claim: Q-020 -- "Does ARC-007's no-value-computation constraint survive MECH-073
(valence intrinsic to the hippocampal map)?"  Resolution A (adopted 2026-04-02 via
literature synthesis) says ARC-007 and MECH-073 are CO-TRUE: the hippocampal map
embodies value-shaped geometry written EXTERNALLY (E3 harm evaluation -> ResidueField
terrain), while HippocampalModule itself computes no value.  This run is the
behavioural test of Resolution A.
Supersedes: V3-EXQ-266 and V3-EXQ-266a
  (run_ids v3_exq_266_q020_valence_geometry_pair_20260410T023257Z_v3,
           v3_exq_266_q020_valence_geometry_pair_20260410T034439Z_v3,
           v3_exq_266a_q020_valence_geometry_pair_fixed_20260411T095750Z_v3).

WHY EXQ-266 AND EXQ-266a WERE INVALIDATED (established 2026-07-22)
------------------------------------------------------------------
Both drivers recovered their executed action from

    argmax(agent.hippocampal.action_object_decoder(
        traj.get_action_object_sequence()[:, 0, :]))

and fed it straight to `env.step` (v3_exq_266_...:269, v3_exq_266a_...:281).  On this
substrate that round trip is a CONSTANT FUNCTION -- it pins to class 3 for every
candidate -- because E2's step-0 action-object embedding is a frozen re-encoding of the
action label with no state dependence, and the resulting logit perturbations (std
0.007-0.017) are an order of magnitude below the decoder's per-class mean gaps (up to
0.33).  See `HippocampalModule.candidate_first_action_class` and ree-v3/CLAUDE.md
"Action-object round trip is NOT an action source + CEM elite floor".

The consequence here is arithmetically forced and visible in the published manifests:
BOTH arms executed the SAME action stream at every seed, because neither the terrain
manipulation nor anything else can reach an action stream that does not depend on the
candidate set.  All three runs report

    harm_rate_TERRAIN_SHAPED = harm_rate_TERRAIN_FLAT = 0.0060999999999999995
    harm_reduction_frac      = 0.0

bit-identical to 16 significant figures.  EXQ-266a is the decisive one: it correctly
FIXED the ablation bug in EXQ-266 (which had gated on `residue.valence_enabled`, a flag
that does not gate scalar `accumulate()`, so both of EXQ-266's arms had identical
terrain -- confirmed by their identical terrain_harm_corr of -0.4384), and its repaired
ablation DID take effect -- terrain_harm_corr moved to -0.4384 SHAPED vs +0.8415 FLAT.
The terrain changed; the behaviour could not.  C1 (harm reduction) was therefore not a
measurement in any of the three runs, and its FAIL is not evidence about Q-020.

Note this is a DIFFERENT signature from the EXQ-114 immobility artifact.  There the
constant action stream inflated one arm's step DENOMINATOR ~14x against a random-walk
control.  Here BOTH arms run the same constant policy, so the denominators are equal
and the artifact hides as a perfect null rather than as a spectacular effect.  Equal
denominators are why the EXQ-114 diagnostic would not have caught it.

WHAT THIS RE-RUN CHANGES (four things, each closing one leg of the artifact)
---------------------------------------------------------------------------
1. SELECTION goes through `agent.select_action(candidates, ticks)` -- E3's J(zeta),
   which returns the action DIRECTLY and never consults `action_object_decoder`.  The
   residue terrain reaches that selection twice over, both PER CANDIDATE: through the
   proposer's terrain prior (`hippocampal/module.py:377` residue_val channel; CEM
   elites are refit on lowest trajectory residue) and through E3's own
   `compute_residue_cost(trajectory)` term in J(zeta)
   (`predictors/e3_selector.py:1020,1046`).
2. Where a candidate's real first action is needed for REPORTING or for the readiness
   gate, it is read with `HippocampalModule.candidate_first_action_class(traj)`, which
   reads `trajectory.actions` -- the ground truth of what the candidate is.  The
   decoder round trip appears nowhere in this file, in either role.
3. A THIRD ARM, A2_STATIC (always action 4 = the env's (0,0) no-op), makes an
   immobility artifact structurally UNABLE to produce a PASS.  C1 requires the shaped
   arm to beat the BETTER of the two controls, so "stand still" is a competitor rather
   than an unexamined confound.
4. The PRIMARY DV is HARM PER EPISODE at a matched episode budget, not harm per step.
   Per-step harm is the statistic an immobility artifact games, so it is recorded as a
   diagnostic and is BARRED FROM EVERY CRITERION.  Episode-termination counts, mean
   episode length, distinct cells visited, distinct executed action classes and the
   cross-arm step-denominator ratio are recorded per arm alongside the DVs.

WHAT THE ABLATION IS, AND WHY IT IS COMPLETE
--------------------------------------------
A1_TERRAIN_FLAT withholds the `residue_field.accumulate()` write on harm; A0 performs
it.  This is EXQ-266a's repair, kept verbatim, and it is the manipulation Q-020 names:
the EXTERNAL write by the harm evaluator is what is supposed to make the map geometry
value-shaped.  The ablation is complete on this driver's call path because the only
other accumulate site, `E3Selector.post_action_update` (e3_selector.py:3297), is
reachable only via `REEAgent.update_residue`, which this loop never calls.  `residue_total`
and `residue_harm_events` are recorded per arm so the ablation is auditable rather than
assumed -- FLAT must land at exactly 0.0 / 0.

WHY A STATIC CONTROL ARM AND NOT A MOBILITY GATE (inherited from V3-EXQ-114a)
-----------------------------------------------------------------------------
Action 4 in `CausalGridWorldV2` is (0,0) -- STAY -- and E3's J(zeta) minimises predicted
harm, so "stand still" is a genuinely attractive policy, not a bug.  A locomotion gate
would misroute a REAL behavioural result ("terrain-guided navigation converges on
immobility") as an instrument failure.  The static control resolves it without a gate:
if the shaped arm converges on stillness it cannot beat A2_STATIC, so it FAILS C1 and
the run reports -- as a measurement -- that value-shaped terrain buys nothing over
standing still.  If it beats both controls, the advantage is real and is not a
denominator artifact, because A2_STATIC has the maximal denominator by construction.

DV-SYMMETRY DECLARATION (mandatory; one line per arm)
-----------------------------------------------------
DV = harm events per eval episode, a function of the realised state trajectory and hence
of the EXECUTED action sequence, which is produced by E3's ranking of J(zeta) across the
hippocampal candidate set.  Symmetry group of that DV: (i) permutation of candidate
index order, (ii) a uniform additive constant across all candidate scores, (iii) any
monotone rescaling of all candidate scores -- none can move the selection.
  A0_TERRAIN_SHAPED  the manipulation is the presence of accumulated residue, which
                     enters BOTH the proposer's terrain prior and E3's J(zeta) as a
                     PER-TRAJECTORY, state-dependent cost (`evaluate_trajectory` over
                     each candidate's own world-state sequence).  It is therefore not a
                     relabelling, not a broadcast constant, and not a monotone map of
                     the whole score vector: not invariant.
  A1_TERRAIN_FLAT    withholds that write entirely, collapsing the residue term to a
                     constant across candidates.  Removing a per-candidate term is not
                     a symmetry of the DV: not invariant.  (Within this arm the residue
                     term IS a broadcast constant -- that is the ablation, by design,
                     and is why the arm's selection is driven by the other J(zeta)
                     terms rather than by terrain.)
  A2_STATIC          replaces selection with a fixed no-op.  A constant policy is not a
                     symmetry of the DV -- it produces a different realised trajectory
                     (the maximal-denominator, zero-locomotion one): not invariant, and
                     that IS its job.
This is precisely the property EXQ-266/266a lacked: their manipulation was invariant
under the executed action stream because that stream was a constant function of nothing.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
-----------------------------------------------------------------------------------
  P1 candidate_action_diversity   A0 and A1: distinct TRUE first-action classes across
                                  the candidate set, read via
                                  candidate_first_action_class (NOT the non-invertible
                                  decoder round trip), measured on a post-training
                                  positive control.  If every candidate proposes the
                                  same action then E3 has nothing to choose between and
                                  the arm is a no-op however selection is routed -- the
                                  EXQ-266 failure mode surviving the selection fix.
                                  Scoped OUT of A2_STATIC, which proposes no candidates
                                  (disposition (a): the arm stays fully scorable).
  P2 terrain_score_spread         A0 only: the SPREAD (max - min) of
                                  residue_field.evaluate(z_world) across the
                                  post-training probe's visited positions.  C5 routes on
                                  a terrain-vs-harm CONTRAST, so per the V3-EXQ-643
                                  same-statistic rule the readiness check asserts a
                                  SPREAD, not a magnitude: a terrain with a large mean
                                  but zero spread carries no positional information and
                                  a contrast measured on it is starved, not falsified.
                                  Scoped OUT of A1_TERRAIN_FLAT (whose terrain is flat
                                  BY CONSTRUCTION -- asserting spread there is
                                  structurally unsatisfiable, disposition (a)) and out
                                  of A2_STATIC.
Per-arm gates are aggregated with experiments/_lib/precondition_gate.py so a red arm can
NEVER vacate a green one (the V3-EXQ-785 whole-run-AND defect).

WHICH CHANNELS EACH READINESS GATE CERTIFIES (the V3-EXQ-604c scope caveat)
---------------------------------------------------------------------------
P1 certifies that the SELECTION channel has something to choose between; it says nothing
about whether the terrain carries positional information.  P2 certifies the TERRAIN
channel and says nothing about candidate diversity.  Neither speaks for the other, and
neither speaks for A2_STATIC, which has no proposal channel at all.

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
--------------------------------------------------------------------
  C1 (LOAD-BEARING) harm_per_episode_shaped <= 0.85 x MIN(flat, static)
                    -- the shaped arm must beat the BETTER of the two controls
  C2 effect >= 0.80 SD of the cross-seed delta vs the BINDING control AND
                    >= an absolute floor of 0.25 harm events per episode
  C3 direction consistent vs the binding control on ALL seeds
  C4 data quality: min harm events per cell >= 20
  C5 terrain_harm_corr in A0 >= 0.10 -- residue terrain scores must predict subsequent
     harm, i.e. value information is present IN the map geometry (MECH-073's half of
     Resolution A).  Required for PASS; not load-bearing, because C1 is the behavioural
     claim and C5 alone could pass on a geometry nothing navigates by.

EXPERIMENT_PURPOSE: evidence (Q-020 is a resolved open question; this supplies the
  behavioural confirmation of Resolution A that EXQ-266/266a failed to measure).

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

EXPERIMENT_TYPE = "v3_exq_266b_q020_valence_geometry_pair"
CLAIM_IDS = ["Q-020"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "v3_exq_266a_q020_valence_geometry_pair_fixed"

# Hold-weighted-E3-readout gate (`validate_experiments.e3_hold_weighted_readout_lint`,
# pseudo-replication defect FORM 2). TRIAGED SAFE by an exact argument, not a bound.
#
# `agent.select_action` returns the HELD action on a non-E3 tick, so `eval_executed` IS
# hold-duration-weighted -- the gate's structural finding is correct. What makes it
# harmless is what that list is consumed by: its ONLY reader is `len(set(...))` (the
# `eval_executed_action_classes` diagnostic). Set CARDINALITY is EXACTLY invariant under
# duplication -- replicating a held class can neither add nor remove a class -- so hold
# weighting cannot move the statistic in either direction, at any E3 cadence. No
# distribution-shape quantity (entropy, variance, histogram mass) is derived from it, and
# it feeds no criterion: C1-C4 route on harm_per_episode, C5 on the terrain contrast, and
# the readiness gates read candidate-set diversity and terrain spread, not this.
#
# Scope caution: the marker is file-wide. A future edit deriving a magnitude or
# distribution-shape statistic from a select_action() return value must re-derive this
# exemption or drop it.
E3_HOLD_WEIGHTED_READOUT_EXEMPT = (
    "eval_executed is consumed ONLY by len(set(...)) (the eval_executed_action_classes "
    "diagnostic); set cardinality is exactly invariant under hold-duration replication, "
    "no distribution-shape statistic is derived from it, and it feeds no criterion"
)

ARM_SHAPED = "A0_TERRAIN_SHAPED"
ARM_FLAT = "A1_TERRAIN_FLAT"
ARM_STATIC = "A2_STATIC"
ARMS = [ARM_SHAPED, ARM_FLAT, ARM_STATIC]

# The env's (0,0) no-op. CausalGridWorldV2.ACTIONS = {0:(-1,0), 1:(1,0), 2:(0,-1),
# 3:(0,1), 4:(0,0)} -- so 4 is STAY, and A2_STATIC is the pure immobility policy.
STATIC_ACTION_IDX = 4

# --- pre-registered thresholds -------------------------------------------
THRESH_C1_REDUCTION = 0.15          # fractional reduction in harm PER EPISODE
THRESH_C2_EFFECT_SD = 0.80          # effect in SD of the cross-seed delta
THRESH_C2_ABS_FLOOR = 0.25          # absolute floor, harm events per episode
THRESH_C4_MIN_HARM_EVENTS = 20      # per cell
THRESH_C5_TERRAIN_CORR = 0.10       # terrain-harm contrast in the shaped arm

# --- non-degeneracy floors ------------------------------------------------
FLOOR_CANDIDATE_ACTION_CLASSES = 1.5   # P1: i.e. >= 2 distinct candidate actions
FLOOR_TERRAIN_SCORE_SPREAD = 1e-4      # P2: terrain must vary with position at all

# Diagnostic-only comparability bound for the per-step rate. NOT a gate: a shaped arm
# that legitimately survives longer inflates this ratio, and that is the success
# signature, not the artifact. EXQ-266/266a's own ratio was exactly 1.00 -- the perfect
# equality that hid their constant action stream.
DIAG_STEP_DENOMINATOR_RATIO_MAX = 3.0

# CEM candidate budget. MUST keep num_elite = int(NUM_CANDIDATES * elite_fraction) at
# >= 2: `HippocampalModule.propose_trajectories` refits with the std over the elite
# stack, and torch's default std() is UNBIASED, so the std of a SINGLE elite is NaN. It
# propagates through the ao_std floor and NaN-poisons every subsequent candidate
# rollout, ending in `RuntimeError: probability tensor contains ... nan` from
# torch.multinomial inside e3_selector.select (ree-v3/CLAUDE.md defect 2). Reachable
# from the substrate default: num_candidates=8 x elite_fraction=0.2 gives int(1.6) == 1.
# 32 is the substrate default and yields num_elite = 6.
NUM_CANDIDATES = 32
MIN_REQUIRED_ELITES = 2
CONTROL_PROBE_STEPS = 25  # post-training positive-control probe for P1 / P2


# =========================================================================
# precondition specs -- each declares the regimes it is meaningful for
# =========================================================================
def _proposes_candidates(ctx: Dict[str, Any]) -> bool:
    """P1 applies only to arms that propose a candidate set at all.

    A2_STATIC executes a fixed no-op and proposes nothing, so a candidate-diversity
    assertion is structurally unsatisfiable there. Disposition (a): scope the
    PRECONDITION out, the arm stays fully scorable.
    """
    return bool(ctx["uses_hippocampal_selection"])


def _terrain_is_shaped(ctx: Dict[str, Any]) -> bool:
    """P2 applies only to the arm whose terrain is supposed to carry information.

    A1_TERRAIN_FLAT's terrain is flat BY CONSTRUCTION -- that is the ablation -- so
    demanding spread there would make the arm structurally un-passable and collapse the
    pair back to one arm. Disposition (a) again: scope the precondition out, not the arm.
    """
    return bool(ctx["terrain_shaped"])


PRECONDITION_SPECS = [
    PreconditionSpec(
        name="candidate_action_diversity",
        description=(
            "distinct TRUE first-action classes across the candidate set, read via "
            "HippocampalModule.candidate_first_action_class (which reads "
            "trajectory.actions, NOT the non-invertible decoder round trip). If every "
            "candidate proposes the same action then E3 has nothing to choose between "
            "and the arm is a no-op however selection is routed -- the EXQ-266 failure "
            "mode surviving the selection fix"
        ),
        control=(
            "post-training positive-control probe over CEM candidate sets that "
            "genuinely differ"
        ),
        threshold=FLOOR_CANDIDATE_ACTION_CLASSES,
        direction="lower",
        applies_to=_proposes_candidates,
        applies_note="only arms that propose a candidate set at all",
    ),
    PreconditionSpec(
        name="terrain_score_spread",
        description=(
            "spread (max - min) of residue_field.evaluate(z_world) across the "
            "post-training probe's visited positions. C5 routes on a terrain-vs-harm "
            "CONTRAST, so the readiness check asserts SPREAD rather than magnitude "
            "(V3-EXQ-643 same-statistic rule): a terrain with a large mean but zero "
            "spread carries no positional information, and a contrast measured on it "
            "is starved, not falsified"
        ),
        control=(
            "post-training probe after the shaped arm has accumulated harm residue"
        ),
        threshold=FLOOR_TERRAIN_SCORE_SPREAD,
        direction="lower",
        applies_to=_terrain_is_shaped,
        applies_note="only the arm whose terrain is shaped by residue accumulation",
    ),
]


def arm_contexts() -> List[Dict[str, Any]]:
    """Pre-registered per-arm context consumed by the precondition gate."""
    return [
        {"id": ARM_SHAPED, "uses_hippocampal_selection": True, "terrain_shaped": True,
         "policy": "e3_over_hippocampal_candidates"},
        {"id": ARM_FLAT, "uses_hippocampal_selection": True, "terrain_shaped": False,
         "policy": "e3_over_hippocampal_candidates"},
        {"id": ARM_STATIC, "uses_hippocampal_selection": False, "terrain_shaped": False,
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
    argmax over `action_object_decoder(get_action_object_sequence()[:, 0, :])` -- is a
    CONSTANT on this substrate and is what invalidated EXQ-266/266a; it appears nowhere
    in this file, in either the selection or the reporting role.
    """
    classes = set()
    for traj in candidates:
        cls = HippocampalModule.candidate_first_action_class(traj)
        if cls is not None:
            classes.add(int(cls))
    return len(classes)


def _terrain_score(agent, z_world: torch.Tensor) -> float:
    try:
        val = float(agent.residue_field.evaluate(z_world).mean().item())
    except Exception:
        return 0.0
    return val if math.isfinite(val) else 0.0


def _select_action(
    agent, latent, ticks, n_actions: int, rng: random.Random,
) -> Tuple[torch.Tensor, Optional[int], bool]:
    """Select via the CANONICAL E3 path over hippocampal terrain-guided candidates.

    `agent.select_action(candidates, ticks)` routes through E3's J(zeta) and returns the
    action DIRECTLY. It never consults `action_object_decoder`, so the executed stream is
    a genuine function of the candidate set -- unlike EXQ-266/266a's round-trip argmax,
    which was a constant function and made BOTH their arms execute one identical stream.

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


def _probe_readiness(
    agent, env, n_actions, rng, n_steps: int,
) -> Dict[str, float]:
    """P1 + P2 positive control, measured AFTER training and BEFORE the scored eval.

    Reports the MINIMUM candidate-class count and the terrain spread over the probe --
    extrema and a range, not means: P1's claim is that the candidate set COULD offer E3 a
    choice (a mean hides the ticks where it could not), and P2's statistic must match the
    spread-shaped contrast C5 routes on.
    """
    counts: List[int] = []
    terrain: List[float] = []
    roundtrip: List[Dict[str, Any]] = []
    _, obs_dict = env.reset()
    agent.reset()
    for _ in range(n_steps):
        with torch.no_grad():
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            ticks = agent.clock.advance()
            terrain.append(_terrain_score(agent, latent.z_world.detach()))
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

    # No candidate set / no terrain sample observed at all -> 0.0, which FAILS the gate
    # closed and routes the arm to substrate_not_ready rather than letting a NaN ride
    # into a comparison that silently evaluates False.
    rt_true = [float(r.get("true_unique_classes") or 0) for r in roundtrip]
    rt_round = [float(r.get("roundtrip_unique_classes") or 0) for r in roundtrip]
    return {
        "candidate_true_action_classes_min": float(min(counts)) if counts else 0.0,
        "candidate_true_action_classes_mean": (
            float(statistics.fmean(counts)) if counts else 0.0
        ),
        "terrain_score_spread": (
            float(max(terrain) - min(terrain)) if len(terrain) > 1 else 0.0
        ),
        "terrain_score_mean_probe": (
            float(statistics.fmean(terrain)) if terrain else 0.0
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
    terrain_shaped = bool(arm_ctx["terrain_shaped"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    # All arms are minted reuse-ELIGIBLE with the driver script EXCLUDED from the hash,
    # so a later, different-driver consumer in the Q-020 / ARC-007 terrain lineage can
    # match them (mint-as-you-go; terminality is unknowable).
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
            reafference_action_dim=0,  # SD-007 off -- isolate the terrain manipulation
        )
        config.latent.unified_latent_mode = False  # SD-005 split latents
        agent = REEAgent(config)
        optimizer = optim.Adam(list(agent.parameters()), lr=full_config["lr"])

        n_warmup = min(3, warmup_episodes) if dry_run else warmup_episodes
        n_eval = min(2, eval_episodes) if dry_run else eval_episodes
        n_steps = min(20, steps_per_episode) if dry_run else steps_per_episode

        # ---------------- TRAIN (E1 + E2; residue accumulates in A0 ONLY) -----
        # THE MANIPULATION: residue_field.accumulate() is gated on terrain_shaped. This
        # is EXQ-266a's repair kept verbatim -- EXQ-266 had gated on
        # residue.valence_enabled, which controls only the SD-014 valence VECTOR and does
        # NOT gate the scalar accumulate, so both its arms had identical terrain.
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
                else:
                    action = _action_to_onehot(
                        STATIC_ACTION_IDX, n_actions, agent.device
                    )
                    agent._last_action = action

                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_train += 1

                if float(harm_signal) < 0:
                    harm_train += 1
                    if terrain_shaped:
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
                    f" residue_total={float(agent.residue_field.total_residue.item()):.4f}",
                    flush=True,
                )

        # ---------------- readiness probe (post-training, pre-eval) ----------
        agent.eval()
        probe = {
            "candidate_true_action_classes_min": 0.0,
            "candidate_true_action_classes_mean": 0.0,
            "terrain_score_spread": 0.0,
            "terrain_score_mean_probe": 0.0,
            "roundtrip_true_unique_classes_mean": 0.0,
            "roundtrip_decoded_unique_classes_mean": 0.0,
        }
        if arm_ctx["uses_hippocampal_selection"]:
            probe = _probe_readiness(
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
        terrain_at_harm: List[float] = []
        terrain_at_safe: List[float] = []
        traj_residue_scores: List[float] = []

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
                    terrain_now = _terrain_score(agent, latent.z_world.detach())

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
                                        traj_residue_scores.append(val)
                    else:
                        action = _action_to_onehot(
                            STATIC_ACTION_IDX, n_actions, agent.device
                        )
                        agent._last_action = action

                eval_executed.append(int(torch.argmax(action, dim=-1).item()))
                _, harm_signal, done, _info, obs_dict = env.step(action)
                steps_eval += 1
                ep_steps += 1
                if float(harm_signal) < 0:
                    harm_eval += 1
                    terrain_at_harm.append(terrain_now)
                else:
                    terrain_at_safe.append(terrain_now)
                ep_cells.add((int(env.agent_x), int(env.agent_y)))
                if done:
                    terminated = True
                    break

            episodes_done += 1
            episodes_terminated += int(terminated)
            episode_lengths.append(ep_steps)
            cells_per_episode.append(len(ep_cells))

        # Terrain-harm contrast: do high-residue positions predict harm?
        terrain_harm_corr = 0.0
        mean_terrain_at_harm = 0.0
        mean_terrain_at_safe = 0.0
        if terrain_at_harm and terrain_at_safe:
            mean_terrain_at_harm = float(statistics.fmean(terrain_at_harm))
            mean_terrain_at_safe = float(statistics.fmean(terrain_at_safe))
            pooled = terrain_at_harm + terrain_at_safe
            std_all = float(statistics.pstdev(pooled)) if len(pooled) > 1 else 0.0
            if std_all > 1e-8:
                terrain_harm_corr = (
                    mean_terrain_at_harm - mean_terrain_at_safe
                ) / std_all

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
            # SECOND DV -- the map-geometry readout (C5)
            "terrain_harm_corr": float(terrain_harm_corr),
            "mean_terrain_at_harm": float(mean_terrain_at_harm),
            "mean_terrain_at_safe": float(mean_terrain_at_safe),
            "n_terrain_harm_samples": int(len(terrain_at_harm)),
            "n_terrain_safe_samples": int(len(terrain_at_safe)),
            # SECONDARY / DIAGNOSTIC -- barred from every criterion (see docstring)
            "harm_per_step": float(harm_per_step),
            "total_steps_eval": int(steps_eval),
            # Denominator instrumentation. EXQ-266/266a's denominators were EQUAL, which
            # is what let the constant action stream hide as a perfect null.
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
            "terrain_score_spread": float(probe["terrain_score_spread"]),
            "terrain_score_mean_probe": float(probe["terrain_score_mean_probe"]),
            "roundtrip_true_unique_classes_mean": float(
                probe["roundtrip_true_unique_classes_mean"]
            ),
            "roundtrip_decoded_unique_classes_mean": float(
                probe["roundtrip_decoded_unique_classes_mean"]
            ),
            # instrument health: steps where no candidate set could be formed and
            # selection fell back to random.
            "hippo_fallback_steps": int(hippo_fallbacks),
            "mean_selected_trajectory_residue": (
                float(statistics.fmean(traj_residue_scores))
                if traj_residue_scores else 0.0
            ),
            "n_residue_samples": int(len(traj_residue_scores)),
            # ABLATION AUDIT: FLAT and STATIC must land at exactly 0.0 / 0.
            "residue_total": float(agent.residue_field.total_residue.item()),
            "residue_harm_events": int(agent.residue_field.num_harm_events.item()),
            "terrain_shaped": bool(terrain_shaped),
        }
        cell.stamp(row)

    print(
        f"  [eval] seed={seed} arm={arm_id}"
        f" harm_per_episode={row['harm_per_episode']:.4f}"
        f" harm_per_step={row['harm_per_step']:.4f}"
        f" terrain_corr={row['terrain_harm_corr']:+.4f}"
        f" mean_ep_len={row['mean_episode_len']:.1f}"
        f" terminated={row['episodes_terminated']}/{row['episodes_done']}"
        f" cells/ep={row['mean_cells_per_episode']:.2f}"
        f" action_classes={row['eval_executed_action_classes']:.0f}"
        f" residue_total={row['residue_total']:.4f}",
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
    corr = {a: _mean([r["terrain_harm_corr"] for r in by_arm[a]]) for a in ARMS}
    steps = {a: sum(r["total_steps_eval"] for r in by_arm[a]) for a in ARMS}

    shaped = hpe[ARM_SHAPED]
    # The BINDING control is whichever trivial/ablated policy is already safer. Q-020
    # requires terrain-guided navigation to beat the BETTER of them: beating only the
    # flat-terrain arm while losing to "stand still" is not evidence that value-shaped
    # map geometry aids behaviour.
    controls = {a: hpe[a] for a in (ARM_FLAT, ARM_STATIC)}
    binding_arm = min(controls, key=lambda a: controls[a])
    binding = controls[binding_arm]
    reduction_frac = float((binding - shaped) / binding) if binding > 0 else 0.0

    # per-seed delta: binding control - shaped (positive = shaped is safer)
    per_seed_delta: List[float] = []
    for s in seeds:
        a0 = next((r for r in by_arm[ARM_SHAPED] if r["seed"] == s), None)
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

    # The EXQ-266/266a signature, made explicit for a later reader: identical DVs across
    # the two proposing arms means the manipulation never reached behaviour.
    arms_bit_identical = bool(
        abs(hpe[ARM_SHAPED] - hpe[ARM_FLAT]) < 1e-12
        and steps[ARM_SHAPED] == steps[ARM_FLAT]
    )

    # Ablation audit -- the manipulation must actually have been applied.
    ablation_clean = bool(
        all(r["residue_total"] > 0.0 for r in by_arm[ARM_SHAPED])
        and all(r["residue_total"] == 0.0 for r in by_arm[ARM_FLAT])
    )

    c1 = bool(reduction_frac >= THRESH_C1_REDUCTION)
    c2 = bool(
        effect_sd >= THRESH_C2_EFFECT_SD and delta_mean >= THRESH_C2_ABS_FLOOR
    )
    c3 = bool(seeds_consistent >= len(seeds))
    c4 = bool(min_harm_events >= THRESH_C4_MIN_HARM_EVENTS)
    c5 = bool(corr[ARM_SHAPED] >= THRESH_C5_TERRAIN_CORR)

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
        # SECOND DV
        "terrain_harm_corr_by_arm": corr,
        # DIAGNOSTIC ONLY -- no criterion reads these
        "harm_per_step_by_arm": hps,
        "total_steps_eval_by_arm": steps,
        "step_denominator_ratio": denom_ratio,
        "harm_per_step_comparable": bool(
            denom_ratio <= DIAG_STEP_DENOMINATOR_RATIO_MAX
        ),
        "arms_bit_identical": arms_bit_identical,
        "ablation_clean": ablation_clean,
        "c1_harm_per_episode_pass": c1,
        "c2_effect_pass": c2,
        "c3_seed_consistency_pass": c3,
        "c4_data_quality_pass": c4,
        "c5_terrain_corr_pass": c5,
        "all_pass": bool(c1 and c2 and c3 and c4 and c5),
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
        if arm_rows:
            # worst cell, not the mean -- a single degenerate cell must not hide behind
            # an in-band average (the recomputability rule).
            if _proposes_candidates(ctx):
                measured["candidate_action_diversity"] = min(
                    r["candidate_true_action_classes_min"] for r in arm_rows
                )
            if _terrain_is_shaped(ctx):
                measured["terrain_score_spread"] = min(
                    r["terrain_score_spread"] for r in arm_rows
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
            "C5_terrain_harm_corr": THRESH_C5_TERRAIN_CORR,
            "P1_floor_candidate_action_classes": FLOOR_CANDIDATE_ACTION_CLASSES,
            "P2_floor_terrain_score_spread": FLOOR_TERRAIN_SCORE_SPREAD,
            "static_action_idx": STATIC_ACTION_IDX,
            "diag_step_denominator_ratio_max": DIAG_STEP_DENOMINATOR_RATIO_MAX,
        },
    }

    # Design-time satisfiability audit: refuse the run before compute is spent if any
    # pre-registered precondition is unsatisfiable from the pre-registered config (the
    # V3-EXQ-785 rule). Never resolve a finding here by lowering a threshold -- that
    # converts a detected artifact into a citable result.
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS,
        arm_contexts(),
    )

    # Same class of design-time proof for the CEM elite floor (ree-v3/CLAUDE.md defect
    # 2): a single elite makes the refit std NaN, poisons every candidate rollout, and
    # crashes torch.multinomial inside e3_selector.select.
    _elite_fraction = float(HippocampalConfig().elite_fraction)
    _num_elite = max(1, int(NUM_CANDIDATES * _elite_fraction))
    if _num_elite < MIN_REQUIRED_ELITES:
        raise ValueError(
            f"CEM refit would use num_elite={_num_elite} from "
            f"num_candidates={NUM_CANDIDATES} x elite_fraction={_elite_fraction}. "
            f"std() over a single elite is NaN, poisons ao_std, NaN-ing every candidate "
            f"rollout and crashing torch.multinomial in e3_selector. Raise "
            f"NUM_CANDIDATES so num_elite >= {MIN_REQUIRED_ELITES}."
        )
    print(
        f"[V3-EXQ-266b] design-audit OK: gate satisfiable, "
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
    degeneracy_reason = str(gate["degeneracy_reason"] or "")

    # A second, independent degeneracy net for the EXACT defect that invalidated
    # EXQ-266/266a: if the two proposing arms produce bit-identical DVs on identical
    # denominators, the manipulation did not reach behaviour and no verdict is available,
    # whatever the readiness gates say.
    if non_degenerate and analysis["arms_bit_identical"]:
        non_degenerate = False
        degeneracy_reason = (
            "A0_TERRAIN_SHAPED and A1_TERRAIN_FLAT produced bit-identical "
            "harm_per_episode on identical step denominators -- the EXQ-266/266a "
            "signature. The terrain manipulation did not reach the executed action "
            "stream, so C1/C2/C3 measured nothing."
        )
    if non_degenerate and not analysis["ablation_clean"]:
        non_degenerate = False
        degeneracy_reason = (
            "ablation audit FAILED: A0 must end with residue_total > 0 and A1 with "
            "residue_total == 0 in every cell. The manipulation was not applied as "
            "pre-registered, so the arms are not the arms this run declared."
        )

    if not non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        interpretation_text = (
            "SUBSTRATE NOT READY -- not a verdict on Q-020. " + degeneracy_reason
        )
    else:
        outcome = "PASS" if analysis["all_pass"] else "FAIL"
        if analysis["all_pass"]:
            label = "q020_resolution_a_behaviourally_confirmed"
            direction = "supports"
            interpretation_text = (
                "Q-020 Resolution A SUPPORTED behaviourally against BOTH controls: "
                "residue terrain written EXTERNALLY by the harm evaluator, navigated by "
                "HippocampalModule and ranked by E3, reduces harm per episode by "
                f"{analysis['harm_per_episode_reduction_frac']*100:.1f}% vs the binding "
                f"control {analysis['binding_control_arm']} "
                f"({analysis['binding_control_harm_per_episode']:.3f} -> "
                f"{analysis['harm_per_episode_by_arm'][ARM_SHAPED]:.3f} harm events per "
                f"episode at a matched episode budget), consistent on "
                f"{analysis['seeds_consistent']}/{len(seeds)} seeds, with the map "
                "geometry itself predicting harm "
                f"(terrain_harm_corr={analysis['terrain_harm_corr_by_arm'][ARM_SHAPED]:+.3f} "
                f">= {THRESH_C5_TERRAIN_CORR}). The advantage comes from TERRAIN "
                "NAVIGATION, not from the hippocampal module evaluating harm: it "
                "computes no value, it navigates geometry someone else wrote. Unlike "
                "EXQ-266/266a this rests on a per-EPISODE denominator, on an action "
                "stream that genuinely depends on the candidate set, and beats a pure "
                "immobility policy."
            )
        else:
            label = "q020_resolution_a_not_behaviourally_confirmed"
            direction = "weakens"
            interpretation_text = (
                "Q-020 Resolution A NOT CONFIRMED behaviourally once selection is routed "
                "through E3, the DV is harm per episode at a matched episode budget, and "
                "a static no-op control is present: reduction vs the binding control "
                f"{analysis['binding_control_arm']} was "
                f"{analysis['harm_per_episode_reduction_frac']*100:.1f}% "
                f"(C1 needs {THRESH_C1_REDUCTION*100:.0f}%), terrain_harm_corr in the "
                f"shaped arm was "
                f"{analysis['terrain_harm_corr_by_arm'][ARM_SHAPED]:+.3f} "
                f"(C5 needs >= {THRESH_C5_TERRAIN_CORR}). The candidate sets offered E3 "
                "a genuine choice and the shaped arm's terrain genuinely varied with "
                "position (gate green), so this is a MEASUREMENT rather than an "
                "instrument failure -- which is exactly what EXQ-266 and EXQ-266a could "
                "not deliver. Read A2_STATIC before concluding anything about "
                "navigation: if it is the binding control, the honest reading is that "
                "standing still is at least as safe as terrain-guided navigation in this "
                "env. Note that Q-020 is RESOLVED on literature (Resolution A, "
                "2026-04-02) and carried no experimental support before this run, so a "
                "negative here weakens the behavioural leg only and does not by itself "
                "reopen the literature ruling."
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
        {"name": "C5_terrain_harm_corr", "load_bearing": False,
         "passed": analysis["c5_terrain_corr_pass"]},
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_arm_gate": gate["per_arm_gate"],
        "interpretation": {
            "label": label,
            "text": interpretation_text,
            "preconditions": gate["adjudication_preconditions"],
            "preconditions_scope_note": gate.get("preconditions_scope_note", ""),
            "criteria_non_degenerate": arm_criteria_non_degenerate(
                {
                    ARM_SHAPED: [
                        "C1_harm_per_episode_reduction",
                        "C2_effect_size",
                        "C3_seed_consistency",
                        "C5_terrain_harm_corr",
                    ],
                    ARM_FLAT: ["C4_data_quality"],
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
        "per_seed_terrain_harm_corr": {
            a: [r["terrain_harm_corr"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "per_seed_mean_episode_len": {
            a: [r["mean_episode_len"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "per_seed_residue_total": {
            a: [r["residue_total"] for r in rows if r["arm_id"] == a] for a in ARMS
        },
        "registered_thresholds": full_config["thresholds"],
        "supersession_note": (
            "Supersedes V3-EXQ-266 and V3-EXQ-266a, whose drivers selected via "
            "argmax(action_object_decoder(get_action_object_sequence()[:,0,:])) -- a "
            "CONSTANT function on this substrate (pins to class 3). Both arms therefore "
            "executed the SAME action stream and all three published runs report "
            "harm_rate_TERRAIN_SHAPED == harm_rate_TERRAIN_FLAT == "
            "0.0060999999999999995 with harm_reduction_frac exactly 0.0. EXQ-266a is "
            "decisive: it repaired EXQ-266's ablation bug (valence_enabled does not gate "
            "scalar accumulate) and its terrain_harm_corr DID move (-0.4384 SHAPED vs "
            "+0.8415 FLAT), proving the manipulation reached the terrain while the "
            "behaviour stayed bit-identical -- an arithmetically forced no-op, not a "
            "null. Mark both manifests evidence_direction: superseded."
        ),
        "harm_per_step_bar_note": (
            "harm_per_step and step_denominator_ratio are recorded for auditability and "
            "are BARRED FROM EVERY CRITERION. Per-step harm is the statistic an "
            "immobility artifact games, so no verdict in this script routes on it. The "
            "denominator ratio is deliberately NOT a gate either: an arm that "
            "legitimately survives longer inflates it, so gating would refuse the "
            "success signature. The artifact is separated from the success by the "
            "A2_STATIC control -- an immobility policy cannot be beaten by an immobility "
            "artifact -- rather than by a locomotion gate, which would misroute a real "
            "'navigation converges on stillness' result as an instrument failure."
        ),
        "predecessor_signature_note": (
            "EXQ-266/266a's signature was NOT the EXQ-114 denominator inflation: both "
            "their arms ran the same constant policy, so their denominators were EQUAL "
            "and the artifact presented as a perfect null. `arms_bit_identical` in the "
            "analysis block is the explicit detector for that shape, and it forces "
            "non_degenerate=false independently of the readiness gates."
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

    print("\n[V3-EXQ-266b] Results", flush=True)
    for a in ARMS:
        print(
            f"  {a}: harm_per_episode="
            f"{analysis['harm_per_episode_by_arm'][a]:.4f}"
            f"  harm_per_step={analysis['harm_per_step_by_arm'][a]:.4f}"
            f"  terrain_corr={analysis['terrain_harm_corr_by_arm'][a]:+.4f}"
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
        f"  harm_per_step_comparable={analysis['harm_per_step_comparable']}"
        f"  arms_bit_identical={analysis['arms_bit_identical']}"
        f"  ablation_clean={analysis['ablation_clean']}",
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
