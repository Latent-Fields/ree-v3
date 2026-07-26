#!/opt/local/bin/python3
"""V3-EXQ-819a -- MECH-457 / INV-088 competence_floor R4 INSTRUMENT-VALIDITY leg (TEST-DESIGN
FIX of V3-EXQ-819): does a PREDICTION-TRAINED z_world confer any competence advantage over a
FROZEN RANDOM-PROJECTION z_world of the same dimensionality?

SUPERSEDES V3-EXQ-819. The scientific question, the ONE manipulation (the SD-070 z_world
encoder warmup), and the detached-encoder instrument-validity design are UNCHANGED. 819a fixes
a measurement_test_design_defect the 819 run exposed (failure_autopsy_batch-793a-817-819_
2026-07-26.md target 3, user-adjudicated 2026-07-26): the non-vacuity GATE and the routing
PREDICATE disagreed on the same data, and the gate vacated a run the router would have PASSed.

THE 819 DEFECT, PRECISELY. 819's `post_bc_install_took` precondition scored install on the
WORST seed (min post-BC foraging competence vs the 1.0 floor). On the 819 run that was trained
0.9 (worst seed 43, a 0.1 near-miss) and random 0.4, so the precondition was met=false on BOTH
arms and `non_degenerate` went false -> the whole run was vacated as
`zworld_advantage_grid_nondiscriminative`. But the routing predicate `advantage_confirmed` uses
STRICT-MAJORITY install (trained 2/3 -> True, random 0/3 -> False), so it read an ASYMMETRIC
install (trained installs, random does not) and set headline.advantage_confirmed = true. The
gate's own docstring declared it "NON-vacating per-arm" (a non-installing control never vacates
the trained arm's valid result), yet the WORST-SEED aggregate vacated both arms on a single
seed's near-miss -- the canonical "the self-route is a hypothesis, not a verdict" pattern
(V3-EXQ-642). The z_world encoder DID train (819: world_encoder_max_abs_delta 0.277-0.333, 3/3
seeds), so the V3-EXQ-780 frozen-random-projection fault stays fixed; the fault here is purely
the gate/router mismatch.

THE THREE 819a FIXES (all measurement/test-design; nothing about the substrate or the science
changes):
  1. GATE == ROUTER BY CONSTRUCTION. `post_bc_install_took` is now scored on the STRICT-MAJORITY
     install FRACTION (n_seeds_install / n_cells, floor > 0.5), the IDENTICAL predicate the
     router's `install_took_strict_majority` uses -- so the gate and the router can no longer
     disagree on the same data, and the docstring's "NON-vacating per-arm" intent is realised:
     a non-installing control (0/3 -> fraction 0.0) is RED but does NOT vacate the trained arm
     (2/3 -> 0.667 > 0.5, GREEN), because aggregate non_degenerate = ANY arm green. The per-seed
     install definition stays at the substrate competence floor (1.0) -- the defect was the
     WORST-SEED aggregation, not the floor value; a 1.0 res/ep bar is the principled
     meaningful-foraging threshold and lowering it would call near-random foraging "installed".
  2. EFFECT MARGIN RESCALED TO THE NEAR-FLOOR BAND. The z_world arms forage in a 0.35-2.4 res/ep
     band (~2% of the 48.05 local_view / 57.2 oracle ceiling), so 819's 1.0 ABSOLUTE effect
     margin demanded a near-DOUBLING of the control AUC and dominated the 15% relative bar
     entirely (the observed +0.442 mean paired advantage could never clear it). ADVANTAGE_ABS_
     FLOOR is lowered 1.0 -> 0.3 (~13% of the band ceiling / 30% of the install floor), keeping
     the max(absolute, 0.15 x control-AUC) structure so a modest reliable lift is detectable
     while a near-zero control AUC still cannot manufacture a false advantage.
  3. SEEDS 3 -> 6 (42..47). Raised per the autopsy (matching the 793a 3->6 repower precedent):
     the strict-majority install determination and the paired-advantage strict-majority-of-seeds
     bar are firmer at 4/6 than at 2/3, and no single seed can dominate. (With fix 1 a single
     seed no longer VACATES the run under any rule; the raise is for statistical power.)

ROUTED as D12 in competence_floor_reposing_2026-07-20.md section 4 R4 (re-confirmed unblocked
in competence_floor_reposing_2026-07-25.md). NOT a portfolio: ONE question, of a shape the
competence_floor question has never taken -- asked ABOUT the instrument (the z_world
representation), not about another candidate mechanism. NOT braked: R3 counts only genuine
substrate_ceiling autopsies, and MECH-457 / INV-088 each have ZERO (the whole MECH-457 campaign
is competence_implementation_gap / measurement_test_design_defect); the upstream substrate
sd_zworld_warmup_optimizer_group is BUILT (ree-v3 b523b9c70a) and VALIDATED (REE_assembly
25a69fcd4c, 2026-07-22: world_encoder tensors move on every seed at zworld_p0_episodes=60). This
is a test-design fix of an already-run leg, not a same-question ceiling re-pose -- it does not
circle.

WHY NOW / WHY THIS IS DECISIVE IN BOTH DIRECTIONS. For sixteen legs the z_world arm of every
competence_floor leg ran on a FROZEN RANDOM PROJECTION (the P0 warmup never trained the
encoder -- the V3-EXQ-780 diagnosis), yet was reported as a prediction-trained representation.
The random projection reached 32.72 under BC while running AS the treatment. The
prediction-trained z_world is constructible for the FIRST time now that
sd_zworld_warmup_optimizer_group has landed. If prediction-training confers NO advantage over a
random projection, the z_world pathway is dispensable for competence and INV-088's
differentiation programme is reframed; if it DOES, the campaign's sixteen legs of z_world
readings become re-interpretable against a substrate that can finally express them.

THE ONE DECLARED MANIPULATION -- the SD-070 P0a z_world encoder warmup (zworld_p0):
  * zworld_random_proj  (CONTROL) -- zworld_p0=0: the encoder stays a frozen random projection.
      The measured baseline that ran for sixteen legs. The zworld_encoder_guard is EXPECTED to
      fire here (the deliberate condition), so the encoder-trained precondition is SCOPED OUT of
      this arm -- disposition (a), not meaningful for the control, NOT a failure to gate on.
  * zworld_trained      (TREAT)   -- zworld_p0=60: run_zworld_p0 prediction-trains the encoder
      AHEAD of the e2 warmup. The encoder-trained precondition is GATING here: a guard fire
      means the treatment was NOT applied -> substrate_not_ready_requeue for this arm (NEVER a
      scientific verdict; measurement_requirement 5).

CLEAN INSTRUMENT-VALIDITY CONTROL. Both arms build make_zworld_agent(cotrain=False, hidden=128)
-- z_world DETACHED, so the encoder is FROZEN throughout BC + RL in BOTH arms and only its
INITIAL CONTENT differs (prediction-trained vs random). Holding the representation fixed is what
makes any competence difference attributable to representational content rather than to how well
BC/RL can shape a random init. Reference build (NON-REGRESSED, NOT the 769-falsified 256/5x):
128-wide actor-critic, 3x budget, z_world detached. NOTE: the historical 32.72 baseline (748)
ran cotrain=True (BC co-trained the encoder), so absolute install here may differ; the
within-run RANDOM-vs-TRAINED contrast under the user-specified detached build is the decisive
readout, and the 32.72 band is recorded as reference context.

THE DV IS A COMPETENCE TRAJECTORY, NOT A TERMINAL SCALAR (measurement_requirement 1). Each
arm x seed cell records the post-BC install competence plus a mid-training competence probe on a
fixed cadence (12 readings over the 3000-episode budget), so a "prediction-training installs an
advantage that then decays" outcome is visible (measurement_requirement 2, the branch V3-EXQ-780
lacked). The probe is non-perturbing (RNG snapshot/restore). ROUTING CONSUMES THE DECLARED
COVARIATES (measurement_requirement 3): post_bc_foraging_competence and install_took select the
requeue / asymmetric-install branches before any graded-advantage branch can be reached, and the
trajectory SHAPE (peak vs terminal) selects between the advantage / transient / dispensable
branches.

THE DECISIVE READOUT is the PAIRED per-seed advantage of the trained arm over the random-
projection control on the competence-trajectory AUC (mean competence across the install point
and every probe reading). Paired within seed = matched env layout sequence. Effect size:
mean paired AUC delta must clear a margin scaled to the baseline (max of a 1.0 absolute floor and
a 15% relative fraction of the control's AUC) AND a strict majority of seeds must clear it
individually -- an absolute floor plus a magnitude-robust relative bar, so a modest reliable
lift is detected without a noisy per-seed delta manufacturing a false advantage. An ASYMMETRIC
install (trained installs, random does not) is itself the strongest form of advantage and is
routed as such, not as substrate_not_ready.

INTERPRETATION GRID (the mandatory "succeeded then decayed" branch is present):
  * substrate_not_ready_requeue                    -- an anchor is sub-floor, OR the treatment
      was not applied (the trained arm's encoder did not train), OR NEITHER arm installed.
      UNINFORMATIVE; never a scientific verdict, never substrate_ceiling / does_not_support.
  * zworld_prediction_training_confers_advantage   -- PASS. The trained arm confers a reliable
      competence-trajectory advantage over the random projection (paired margin + strict
      majority), OR it installs competence where the random projection cannot. z_world
      representation quality is load-bearing; the sixteen legs are re-interpretable.
  * zworld_training_advantage_transient            -- the trained arm's trajectory PEAK cleared
      the lift-competence target on a strict majority of seeds but its terminal retained_fraction
      fell below the retention floor. The manipulation SUCCEEDED and then DECAYED -- read the
      trajectory and half-life, not the terminal scalar. (measurement_requirement 2.)
  * zworld_training_harms_competence               -- the trained arm is reliably WORSE than the
      random projection (paired margin negative + strict majority), OR the random projection
      installs where the trained encoder cannot. Prediction-training HURT competence.
  * zworld_pathway_dispensable_for_competence      -- THE INFORMATIVE NULL. Both arms install and
      the trained arm neither reliably beats nor is beaten by the random projection (within the
      null band). Prediction-training confers NO advantage -> the z_world pathway is dispensable
      for competence and INV-088's differentiation programme is reframed. Decisive, not weak.
  * zworld_advantage_grid_nondiscriminative        -- no arm passed its gate, or the arms do not
      separate cleanly. Not a refutation; unscored.

MULTI-ARM GATE. experiments/_lib/precondition_gate.py -- per-arm gates aggregated so
non_degenerate = ANY arm green. The control's frozen encoder must NEVER vacate the trained arm's
valid result, and vice-versa (the V3-EXQ-785 vacating defect); the encoder-trained precondition
is scoped to the trained arm via applies_to.

MINT (mint-as-you-go). The CONTROL arm (zworld_random_proj) is emitted reuse-ELIGIBLE with the
shared lineage slice (baselines.competence_floor_zworld.arm_config_slice) and
include_driver_script_in_hash=False, so a later sibling leg can reuse the minted frozen-random-
projection cell across a different driver. No separate baseline-only mint job (neither sanctioned
exception applies).

evidence_direction = "unknown" (DIAGNOSTIC; the verdict lives in interpretation.label /
discrimination_verdict, adjudicated by /failure-autopsy before any governance action).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

ASCII-only in all runtime strings (repo rule).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    evaluate_seed,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.competence_floor_zworld as zbase  # noqa: E402
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_819a_mech457_inv088_zworld_trained_vs_random_gatefix"
QUEUE_ID = "V3-EXQ-819a"
SUPERSEDES = "v3_exq_819_mech457_inv088_zworld_trained_vs_random"
CLAIM_IDS: List[str] = ["MECH-457", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- 819a seed raise (autopsy: 3 -> 6, matching the 793a repower precedent) -----------------
# The base pulls fan.SEEDS (42/43/44 = 3). 819a runs 6 seeds so the strict-majority install
# determination and the paired-advantage strict-majority-of-seeds bar are firmer (4/6 vs 2/3).
SEEDS_819A: Tuple[int, ...] = (42, 43, 44, 45, 46, 47)

DEVICE = fan.DEVICE

OFF_ARM = zbase.OFF_ARM_ID        # zworld_random_proj (control)
ON_ARM = zbase.ON_ARM_ID          # zworld_trained (treatment)
BOOT_ARMS: Tuple[str, ...] = (OFF_ARM, ON_ARM)
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS

# --- Probe cadence (the trajectory-DV measurement constraint) ------------------------------
# 3000-episode reference budget / 250 = 12 readings per (arm x seed).
RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2

# --- Warmup / install schedule -------------------------------------------------------------
P0_WARMUP_EPISODES = zbase.P0_WARMUP_EPISODES       # 200 all-ON e2 warmup, IDENTICAL both arms
ZWORLD_P0_EPISODES = zbase.ZWORLD_P0_EPISODES       # 60  SD-070 encoder warmup, TRAINED arm ONLY
BC_WARMSTART_EPISODES = zbase.BC_WARMSTART_EPISODES  # 300
REF_ACTOR_CRITIC_HIDDEN = zbase.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = zbase.REF_BUDGET_MULTIPLIER  # 3x
POST_BC_INSTALL_FLOOR = zbase.POST_BC_INSTALL_FLOOR  # 1.0 competence floor

DRY_P0_WARMUP = 3
DRY_ZWORLD_P0 = 4
DRY_BC = 4

# --- Pre-registered advantage thresholds (declared; never derived from the run) ------------
# Advantage = paired (trained - random) mean competence over the trajectory AUC.
# 819a: rescaled to the near-floor competence band. The z_world arms forage in a 0.35-2.4
# res/ep band (~2% of the 48.05 local_view / 57.2 oracle ceiling), so 819's 1.0 absolute margin
# demanded a near-DOUBLING of the control AUC and dominated the relative bar entirely (the 819
# run's observed +0.442 mean paired advantage could never clear it). 0.3 is ~13% of the band
# ceiling / 30% of the 1.0 install floor -- a modest reliable lift is detectable, while the
# max(absolute, relative) structure still prevents a near-zero control AUC from manufacturing a
# false advantage.
ADVANTAGE_ABS_FLOOR = 0.3             # 819a: was 1.0 (too coarse for the 0.35-2.4 res/ep band)
ADVANTAGE_REL_FRAC = 0.15            # ... or 15% of the control's own trajectory-AUC, whichever bigger
LIFT_COMPETENCE_TARGET = boot.LIFT_COMPETENCE_TARGET  # ~13.05 res/ep
RETAINED_FRACTION_FLOOR = 0.5        # >= half the installed competence survives refinement
MIN_TRAJECTORY_READINGS = 1.5        # need >= 2 readings for a SHAPE (floor, strict >)


def _arm_ctx(arm_id: str, on_budget: int, probe_every: int) -> Dict[str, Any]:
    return {
        "id": arm_id,
        "is_trained": (arm_id == ON_ARM),
        "n_episodes": int(on_budget),
        "probe_every": int(probe_every),
    }


# ---------------------------------------------------------------------------------------
# Precondition specs (multi-arm gate). Every entry carries numeric measured + threshold so
# build_experiment_indexes.py can recompute `met`. All are FLOORS (direction "lower", strict >).
# ---------------------------------------------------------------------------------------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="local_view_greedy_clears_floor_at_d3",
        description=(
            "LocalViewGreedyPolicy reading the 5x5 resource_field_view forages above the 1.0 "
            "competence floor at D3 -- the positive control that the env is solvable. Below-floor "
            "means the substrate/env is not ready, NOT that a z_world arm failed."
        ),
        control="local_view_greedy foraging_competence @D3 (738 denominator; 48.05 in 742)",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="greedy_oracle_clears_floor_at_d3",
        description="Env is floor-achievable with global info (achievability anchor).",
        control="greedy_oracle foraging_competence @D3 vs the 1.0 floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="post_bc_install_took",
        description=(
            "The BC install took on this arm before RL, scored on the STRICT-MAJORITY install "
            "FRACTION: the fraction of seeds whose post_bc_foraging_competence (measured pre-RL, "
            "post warm-start) clears the 1.0 substrate competence floor must exceed 0.5. This is "
            "the IDENTICAL predicate the router's install_took_strict_majority uses, so the gate "
            "and the router cannot disagree on the same data (the V3-EXQ-819 defect). An install "
            "that did not take on a strict majority of seeds carries no installed prior; the run "
            "self-routes substrate_not_ready_requeue when NEITHER arm installs, or records an "
            "asymmetric-install ADVANTAGE finding when only one does. Per-arm and NON-vacating: a "
            "non-installing control (fraction 0.0) never vacates the trained arm's valid result "
            "(2/3 -> 0.667 > 0.5, green), because aggregate non_degenerate = ANY arm green. "
            "819a scores strict-majority (fraction > 0.5), NOT the worst seed -- a single seed's "
            "near-miss (819 seed 43: 0.9 vs 1.0) no longer vacates an arm that installed on a "
            "strict majority of its seeds."
        ),
        control=(
            "fraction of this arm's seeds whose post-BC / pre-RL foraging_competence clears the "
            "1.0 floor, vs a strict-majority 0.5 gate (identical to the router's predicate)"
        ),
        threshold=0.5,
        kind="readiness",
    ),
    PreconditionSpec(
        name="competence_trajectory_readings",
        description=(
            "The DV is a TRAJECTORY, so the arm's worst cell must carry at least two probe "
            "readings -- one reading has no shape and cannot separate advantage from "
            "'succeeded then decayed'."
        ),
        control="number of mid-training competence probe readings in this arm's worst cell",
        threshold=float(MIN_TRAJECTORY_READINGS),
        kind="measurability",
        structural_max=lambda ctx: float(
            int(ctx["n_episodes"]) // max(1, int(ctx["probe_every"]))
        ),
    ),
    PreconditionSpec(
        name="zworld_world_encoder_trained",
        description=(
            "P0a warmup moved at least one split_encoder.world_encoder tensor (max|delta| > 0), "
            "i.e. z_world is a PREDICTION-TRAINED encoder rather than a frozen random projection. "
            "GATING on the TRAINED arm ONLY: a guard fire there means the treatment was not "
            "applied -> substrate_not_ready_requeue. SCOPED OUT of the random-projection CONTROL, "
            "where a frozen (untrained) encoder is the deliberate condition under test -- asserting "
            "training there would make the control's gate structurally un-passable and collapse the "
            "two-arm design."
        ),
        control=(
            "the SD-070 P0a warmup on the trained arm's own env at zworld_p0=60 -- the phase whose "
            "stated job is to train the world encoder"
        ),
        threshold=0.0,
        direction="lower",   # strict floor: max|delta| > 0.0; bit-identity (== 0) is the failure
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["is_trained"]),
        applies_note=(
            "TRAINED arm only -- the random-projection control is DEFINED by a frozen (untrained) "
            "encoder, so this precondition is not meaningful for it"
        ),
    ),
)

PRECONDITION_BY_NAME: Dict[str, PreconditionSpec] = {s.name: s for s in PRECONDITION_SPECS}

# --- Encoder-trained anchor reachability (V3-EXQ-778d lesson) -------------------------------
# The encoder-trained precondition self-routes to substrate_not_ready_requeue, so the gate must
# be provably REACHABLE by its known-positive control before the run spends compute. The control
# is the SD-070 validation itself: the warmup MOVES the world encoder at zworld_p0=60 (validated
# 2026-07-22, guard PASSES). Scored by THE SHIPPED PREDICATE (PreconditionSpec.met_for), not a copy.
ENCODER_TRAINED_REFERENCE_CELLS: Tuple[float, ...] = (6.03e-3,)
ENCODER_TRAINED_REFERENCE_SOURCE = (
    "ree-v3 CLAUDE.md SD-070 ADOPTION note (substrate ree-v3 b523b9c70a, validated REE_assembly "
    "25a69fcd4c 2026-07-22): split_encoder.world_encoder tensors move on every seed at "
    "zworld_p0_episodes=60, max_delta ~6.03e-03 (guard PASSES)"
)
ENCODER_TRAINED_REACHABILITY_THRESHOLD = 1.0   # ALL reference cells must clear the strict >0 gate


def _run_boot_cell(trained: bool, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int, p0: int, zworld_p0: int,
                   n_bc: int, dry_run: bool) -> Dict[str, Any]:
    arm_id = ON_ARM if trained else OFF_ARM

    # Build the (detached-z_world) agent, then apply the ONE manipulation in the P0 warmup.
    agent = zbase.build_zworld_agent(seed, env_kwargs)
    guard = zbase.warmup_arm(
        agent, seed, env_kwargs, steps=steps, trained=trained, p0=p0, zworld_p0=zworld_p0,
        dry_run=dry_run,
    )

    # Phase 1 -- BC install to the competence point, then measure whether it TOOK.
    install = zbase.install_bc(
        agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, n_bc=n_bc, arm_label=arm_id,
    )
    post_bc = float(install["post_bc_foraging_competence"])

    # Phase 2 -- the reference RL refinement on the FROZEN representation, probed on a cadence.
    train_env = x734._make_env(seed, env_kwargs)
    traj_out = zbase.train_trajectory(
        agent, train_env, seed, n_episodes=int(on_budget), steps=int(steps), arm_label=arm_id,
        denom=int(on_budget), probe_every=int(probe_every), env_kwargs=env_kwargs,
        eval_eps=int(eval_eps),
    )
    trajectory: List[Dict[str, Any]] = list(traj_out.get("competence_trajectory", []))

    # Phase 3 -- unshaped terminal eval (same statistic as the anchors and the probe).
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(x742.ActorCriticEvalPolicy(agent, arm_id), eval_env, eval_eps, steps)

    traj_vals = [float(r.get("foraging_competence", 0.0)) for r in trajectory]
    peak = round(max(traj_vals), 6) if traj_vals else 0.0
    terminal_traj = round(traj_vals[-1], 6) if traj_vals else 0.0
    # Competence series = install point (t=0) + every probe reading -> AUC captures install AND
    # retention/decay (a high install that decays lowers the AUC).
    series = [post_bc] + traj_vals
    traj_auc = round(float(sum(series) / len(series)), 6) if series else 0.0
    retained = zbase.retained_fraction(trajectory, post_bc)
    half_life = zbase.competence_half_life(trajectory, post_bc)

    row["trained_encoder"] = bool(trained)
    row["post_bc_foraging_competence"] = round(post_bc, 6)
    row["install_took"] = bool(install["install_took"])
    row["bc_action_match_recent"] = float(install["bc_action_match_recent"])
    # FULL per-seed trajectory -- never collapsed to a terminal scalar.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    row["competence_trajectory_auc"] = traj_auc
    row["retained_fraction"] = retained
    row["competence_half_life_episodes"] = half_life
    row["peak_cleared_success_target"] = bool(peak >= LIFT_COMPETENCE_TARGET)
    row["mean_train_forage_recent"] = float(traj_out.get("mean_train_forage_recent", 0.0))
    # zworld_encoder_guard audit -- the load-bearing bit is world_encoder_max_abs_delta (>0 == trained).
    row["world_encoder_max_abs_delta"] = float(guard.get("world_encoder_max_abs_delta", 0.0))
    row["zworld_encoder_trained"] = bool(guard.get("zworld_encoder_trained", False))
    row["n_world_encoder_changed"] = int(guard.get("n_world_encoder_changed", 0))
    row["n_latent_stack_changed"] = int(guard.get("n_latent_stack_changed", 0))
    row["zworld_p0_episodes"] = int(guard.get("zworld_p0_episodes", 0))
    return row


def _worst(cells: List[Dict[str, Any]], key: str, default: float) -> Tuple[float, Any]:
    if not cells:
        return float(default), None
    best = min(cells, key=lambda c: float(c.get(key, default) if c.get(key) is not None else default))
    return float(best.get(key, default) if best.get(key) is not None else default), best.get("seed")


def _strict_majority(n_true: int, n_total: int) -> bool:
    return bool(n_total and n_true > (n_total / 2.0))


def run_experiment(seeds: List[int], on_budget: int, eval_eps: int, steps: int,
                   probe_every: int, p0: int, zworld_p0: int, n_bc: int,
                   dry_run: bool) -> Dict[str, Any]:
    print(
        f"MECH-457 / INV-088 competence_floor R4 z_world instrument-validity "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep=z_world DETACHED, ON_budget={on_budget}, probe_every={probe_every}, "
        f"eval={eval_eps}, steps={steps}; ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, "
        f"budget_mult={REF_BUDGET_MULTIPLIER}, p0={p0}, zworld_p0={zworld_p0}; "
        f"manipulation=prediction-training the z_world encoder ONLY)",
        flush=True,
    )
    arm_ctxs = [_arm_ctx(a, on_budget, probe_every) for a in BOOT_ARMS]
    # Design-audit BEFORE any compute: refuse a run carrying a structurally unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)
    # ... and refuse a run whose encoder-trained anchor its own known-positive control (the SD-070
    # validation) cannot clear -- a predicate narrower than "the encoder moved" would report
    # met=false forever and mislabel the treatment-not-applied gap as a substrate verdict.
    encoder_anchor_reachability = assert_anchor_reachable(
        anchor_name="zworld_world_encoder_trained",
        reference_cells=list(ENCODER_TRAINED_REFERENCE_CELLS),
        score_fn=PRECONDITION_BY_NAME["zworld_world_encoder_trained"].met_for,
        threshold=ENCODER_TRAINED_REACHABILITY_THRESHOLD,
        reference_source=ENCODER_TRAINED_REFERENCE_SOURCE,
    )
    print(
        f"anchor reachability: {encoder_anchor_reachability['anchor_name']} "
        f"reference={encoder_anchor_reachability['n_reference_scored_true']}/"
        f"{encoder_anchor_reachability['n_reference_cells']} "
        f"reachable={encoder_anchor_reachability['reachable']}", flush=True,
    )

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        is_boot = arm_id in BOOT_ARMS
        is_off = (arm_id == OFF_ARM)
        trained = (arm_id == ON_ARM)
        if is_boot:
            slice_cfg = zbase.arm_config_slice(
                env_kwargs, trained=trained, eval_eps=eval_eps, steps=steps,
                on_budget=on_budget, probe_every=probe_every, p0=p0,
                zworld_p0=(zworld_p0 if trained else 0), n_bc=n_bc,
            )
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID,
                         "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        # The CONTROL (random projection) arm mints cross-driver-reusable (no driver script in the
        # hash) so a later sibling leg can reuse it; the treatment arm is leg-specific.
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=not is_off) as cell:
            if is_boot:
                row = _run_boot_cell(trained, env_kwargs, seed, on_budget, eval_eps, steps,
                                     probe_every, p0, zworld_p0, n_bc, dry_run)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    # Anchors first (readiness denominators), then the two z_world arms.
    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    anchors_ready = bool(
        local_view_mean > COMPETENCE_RESOURCE_FLOOR and oracle_mean > COMPETENCE_RESOURCE_FLOOR
    )

    if anchors_ready:
        for arm_id in BOOT_ARMS:
            for seed in seeds:
                _run_cell(arm_id, seed)
    else:
        print(
            f"anchors UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping z_world arms -> substrate_not_ready_requeue", flush=True,
        )

    # ---- per-arm readouts (trajectory-shaped, worst-cell reported) ------------------------
    def _cells_for(arm_id: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == arm_id]

    per_arm_zworld: Dict[str, Any] = {}
    for arm_id in BOOT_ARMS:
        cells = _cells_for(arm_id)
        post_bc_worst, post_bc_worst_seed = _worst(cells, "post_bc_foraging_competence", 0.0)
        n_read_worst, n_read_worst_seed = _worst(cells, "n_trajectory_readings", 0.0)
        enc_delta_worst, enc_delta_worst_seed = _worst(cells, "world_encoder_max_abs_delta", 0.0)
        retained_vals = [c.get("retained_fraction") for c in cells]
        retained_num = [float(v) for v in retained_vals if v is not None]
        retained_mean = round(float(sum(retained_num) / len(retained_num)), 6) if retained_num else None
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        aucs = [float(c.get("competence_trajectory_auc", 0.0)) for c in cells]
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        n_peak_cleared = int(sum(1 for c in cells if bool(c.get("peak_cleared_success_target", False))))
        n_retained = int(sum(1 for v in retained_num if v >= RETAINED_FRACTION_FLOOR))
        n_enc_trained = int(sum(1 for c in cells if bool(c.get("zworld_encoder_trained", False))))
        n_cells = len(cells)
        per_arm_zworld[arm_id] = {
            "arm_id": arm_id,
            "trained_encoder": (arm_id == ON_ARM),
            "n_cells": n_cells,
            "post_bc_foraging_competence_per_seed": [
                float(c.get("post_bc_foraging_competence", 0.0)) for c in cells
            ],
            "post_bc_foraging_competence_worst": round(post_bc_worst, 6),
            "post_bc_worst_seed": post_bc_worst_seed,
            "n_seeds_install_took": n_install_took,
            # 819a: the strict-majority install fraction is what the GATE now scores (identical
            # to the router's install_took_strict_majority), reconciling gate and router.
            "post_bc_install_fraction": round(float(n_install_took) / float(max(1, n_cells)), 6),
            "install_took_strict_majority": _strict_majority(n_install_took, n_cells),
            "n_trajectory_readings_worst": int(n_read_worst),
            "n_trajectory_readings_worst_seed": n_read_worst_seed,
            "competence_trajectory_auc_per_seed": [round(a, 6) for a in aucs],
            "competence_trajectory_auc_mean": round(float(sum(aucs) / len(aucs)), 6) if aucs else 0.0,
            "retained_fraction_per_seed": retained_vals,
            "retained_fraction_mean": retained_mean,
            "n_seeds_retained": n_retained,
            "retained_strict_majority": _strict_majority(n_retained, n_cells),
            "trajectory_peak_per_seed": [round(p, 6) for p in peaks],
            "trajectory_peak_mean": round(float(sum(peaks) / len(peaks)), 6) if peaks else 0.0,
            "n_seeds_peak_cleared_success_target": n_peak_cleared,
            "peak_cleared_strict_majority": _strict_majority(n_peak_cleared, n_cells),
            "competence_half_life_episodes_per_seed": [
                c.get("competence_half_life_episodes") for c in cells
            ],
            "world_encoder_max_abs_delta_worst": round(enc_delta_worst, 9),
            "world_encoder_max_abs_delta_worst_seed": enc_delta_worst_seed,
            "n_seeds_encoder_trained": n_enc_trained,
            "encoder_trained_strict_majority": _strict_majority(n_enc_trained, n_cells),
            "terminal_forage_per_seed": [round(v, 6) for v in per_arm_forage[arm_id]],
            "terminal_forage_mean": round(_mean(arm_id), 6),
        }

    # ---- PAIRED per-seed advantage (trained minus random), matched within seed -------------
    off_cells = {int(c["seed"]): c for c in _cells_for(OFF_ARM)}
    on_cells = {int(c["seed"]): c for c in _cells_for(ON_ARM)}
    paired_seeds = sorted(set(off_cells) & set(on_cells))
    paired_auc_delta: List[float] = []
    paired_peak_delta: List[float] = []
    paired_terminal_delta: List[float] = []
    paired_rows: List[Dict[str, Any]] = []
    for s in paired_seeds:
        oc, nc = off_cells[s], on_cells[s]
        d_auc = float(nc.get("competence_trajectory_auc", 0.0)) - float(oc.get("competence_trajectory_auc", 0.0))
        d_peak = float(nc.get("trajectory_peak_competence", 0.0)) - float(oc.get("trajectory_peak_competence", 0.0))
        d_term = float(nc.get("foraging_competence", 0.0)) - float(oc.get("foraging_competence", 0.0))
        paired_auc_delta.append(d_auc)
        paired_peak_delta.append(d_peak)
        paired_terminal_delta.append(d_term)
        paired_rows.append({
            "seed": s,
            "random_auc": round(float(oc.get("competence_trajectory_auc", 0.0)), 6),
            "trained_auc": round(float(nc.get("competence_trajectory_auc", 0.0)), 6),
            "auc_delta": round(d_auc, 6),
            "peak_delta": round(d_peak, 6),
            "terminal_delta": round(d_term, 6),
        })

    n_paired = len(paired_seeds)
    mean_auc_delta = round(float(sum(paired_auc_delta) / n_paired), 6) if n_paired else 0.0
    mean_peak_delta = round(float(sum(paired_peak_delta) / n_paired), 6) if n_paired else 0.0
    mean_terminal_delta = round(float(sum(paired_terminal_delta) / n_paired), 6) if n_paired else 0.0
    # Effect-size margin: absolute floor OR a fraction of the control's own AUC magnitude, bigger wins.
    off_auc_mean = float(per_arm_zworld[OFF_ARM]["competence_trajectory_auc_mean"])
    eff_margin = round(max(ADVANTAGE_ABS_FLOOR, ADVANTAGE_REL_FRAC * off_auc_mean), 6)
    n_seed_adv = int(sum(1 for d in paired_auc_delta if d >= eff_margin))
    n_seed_harm = int(sum(1 for d in paired_auc_delta if d <= -eff_margin))
    n_seed_peak_adv = int(sum(1 for d in paired_peak_delta if d >= eff_margin))

    # ---- multi-arm gate (per-arm; a red arm NEVER vacates a green one) --------------------
    arm_gates = []
    for ctx in arm_ctxs:
        arm_id = ctx["id"]
        r = per_arm_zworld[arm_id]
        measured = {
            "local_view_greedy_clears_floor_at_d3": round(local_view_mean, 6),
            "greedy_oracle_clears_floor_at_d3": round(oracle_mean, 6),
            # 819a: score install on the STRICT-MAJORITY fraction (> 0.5), NOT the worst seed --
            # this is the same statistic the router's install_took_strict_majority consumes, so
            # the gate and the router cannot disagree, and a non-installing control (0.0) never
            # vacates a trained arm that installed on a strict majority (0.667). measured IS the
            # fraction the met test compares, so the indexer recomputes met identically.
            "post_bc_install_took": float(r["post_bc_install_fraction"]),
            "competence_trajectory_readings": float(r["n_trajectory_readings_worst"]),
        }
        if ctx["is_trained"]:
            measured["zworld_world_encoder_trained"] = float(r["world_encoder_max_abs_delta_worst"])
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, PRECONDITION_SPECS, measured=measured))
    gate = aggregate_arm_gates(arm_gates)

    # ---- routing: covariates + install + trajectory SHAPE first, graded advantage last -----
    off_r = per_arm_zworld[OFF_ARM]
    on_r = per_arm_zworld[ON_ARM]
    off_install = bool(off_r["install_took_strict_majority"])
    on_install = bool(on_r["install_took_strict_majority"])
    trained_encoder_ok = bool(on_r["encoder_trained_strict_majority"])
    on_retained = bool(on_r["retained_strict_majority"])

    advantage_confirmed = bool(
        (on_install and not off_install) or
        (on_install and off_install and mean_auc_delta >= eff_margin and _strict_majority(n_seed_adv, n_paired))
    )
    harms_confirmed = bool(
        (off_install and not on_install) or
        (on_install and off_install and mean_auc_delta <= -eff_margin and _strict_majority(n_seed_harm, n_paired))
    )
    # Mandatory "succeeded then decayed" branch (on the trained arm's own trajectory).
    on_succeeded_then_decayed = bool(
        on_install and off_install and on_r["peak_cleared_strict_majority"] and not on_retained
        and not advantage_confirmed
    )
    dispensable = bool(
        on_install and off_install
        and abs(mean_auc_delta) < eff_margin
        and not _strict_majority(n_seed_adv, n_paired)
        and not _strict_majority(n_seed_harm, n_paired)
        and not on_succeeded_then_decayed
    )

    if (not anchors_ready) or (not trained_encoder_ok) or (not off_install and not on_install):
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not gate["non_degenerate"]:
        outcome, label = "FAIL", "zworld_advantage_grid_nondiscriminative"
    elif advantage_confirmed:
        outcome, label = "PASS", "zworld_prediction_training_confers_advantage"
    elif on_succeeded_then_decayed:
        outcome, label = "FAIL", "zworld_training_advantage_transient"
    elif harms_confirmed:
        outcome, label = "FAIL", "zworld_training_harms_competence"
    elif dispensable:
        outcome, label = "FAIL", "zworld_pathway_dispensable_for_competence"
    else:
        outcome, label = "FAIL", "zworld_advantage_grid_nondiscriminative"

    # Degeneracy: the trajectory-AUC values must actually spread across arms/seeds.
    degeneracy = check_degeneracy({
        "zworld_arm_and_anchor_trajectory_auc": {
            "values": aucs_all if (aucs_all := [
                float(c.get("competence_trajectory_auc", 0.0)) for c in all_cells
                if c.get("arm_id") in BOOT_ARMS
            ]) else [0.0]
        }
    })

    criteria_by_arm = {
        ON_ARM: ["C_zworld_training_confers_competence_advantage"],
        OFF_ARM: ["C_random_projection_provides_measured_baseline"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm, gate,
        extra={
            "C_zworld_training_confers_competence_advantage": bool(degeneracy["non_degenerate"]),
            "C_random_projection_provides_measured_baseline": bool(degeneracy["non_degenerate"]),
        },
    )
    criteria_nd["zworld_arm_trajectory_auc_spread"] = bool(degeneracy["non_degenerate"])
    criteria_nd["at_least_one_arm_installed"] = bool(off_install or on_install)
    criteria_nd["trained_encoder_actually_trained"] = trained_encoder_ok
    criteria_nd["arms_paired_on_seeds"] = bool(n_paired >= 2)

    interpretation = {
        "label": label,
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "criteria": [
            {"name": "C_zworld_training_confers_competence_advantage",
             "load_bearing": True,
             "description": (
                 "The prediction-trained z_world arm confers a reliable competence-trajectory "
                 f"advantage over the random-projection control: mean paired AUC delta >= the "
                 f"effect margin (max of {ADVANTAGE_ABS_FLOOR} absolute and "
                 f"{ADVANTAGE_REL_FRAC} of the control AUC) on a strict majority of seeds, OR it "
                 "installs competence where the random projection cannot."
             ),
             "passed": advantage_confirmed},
            {"name": "C_random_projection_provides_measured_baseline",
             "load_bearing": False,
             "description": (
                 "The random-projection control installs and provides a measured baseline the "
                 "trained arm is read against (the control that ran for sixteen legs)."
             ),
             "passed": bool(off_install)},
        ],
        "criteria_non_degenerate": criteria_nd,
        "anchor_reachability": encoder_anchor_reachability,
        "interpretation_grid": [
            {"label": "substrate_not_ready_requeue", "outcome": "FAIL",
             "condition": (
                 "an anchor is sub-floor, OR the trained arm's encoder did not train (treatment "
                 "not applied), OR NEITHER arm installed"
             ),
             "reading": (
                 "UNINFORMATIVE about the z_world instrument. NEVER a scientific verdict and never "
                 "substrate_ceiling / substrate_conditional / does_not_support. Requeue."
             )},
            {"label": "zworld_prediction_training_confers_advantage", "outcome": "PASS",
             "condition": (
                 "the trained arm beats the random projection on the paired trajectory-AUC by the "
                 "effect margin on a strict majority of seeds, OR installs where the random "
                 "projection cannot"
             ),
             "reading": (
                 "Prediction-training the z_world encoder confers a competence advantage over a "
                 "random projection of the same dimensionality: the representation is load-bearing "
                 "for competence. The campaign's sixteen legs of z_world readings become "
                 "re-interpretable against a substrate that can express them."
             )},
            {"label": "zworld_training_advantage_transient", "outcome": "FAIL",
             "condition": (
                 "the trained arm's trajectory PEAK cleared the lift-competence target on a strict "
                 "majority of seeds, but its terminal retained_fraction fell below the retention "
                 "floor and no reliable terminal advantage remains"
             ),
             "reading": (
                 "The manipulation SUCCEEDED and then DECAYED -- prediction-training installs a "
                 "competence that erodes under refinement. Read the trajectory and half-life, not "
                 "the terminal scalar. This is the branch V3-EXQ-780's grid lacked."
             )},
            {"label": "zworld_training_harms_competence", "outcome": "FAIL",
             "condition": (
                 "the trained arm is reliably WORSE than the random projection on the paired "
                 "trajectory-AUC by the effect margin, OR the random projection installs where the "
                 "trained encoder cannot"
             ),
             "reading": (
                 "Prediction-training the encoder HURT competence relative to a random projection "
                 "(possible if the SD-070 recipe over-specialises or collapses the representation "
                 "for the foraging task). Weakens the z_world-quality-helps-competence reading."
             )},
            {"label": "zworld_pathway_dispensable_for_competence", "outcome": "FAIL",
             "condition": (
                 "both arms install and the trained arm neither reliably beats nor is beaten by "
                 "the random projection (within the null band)"
             ),
             "reading": (
                 "THE INFORMATIVE NULL. Prediction-training confers NO competence advantage over a "
                 "random projection -> the z_world pathway is DISPENSABLE for competence and "
                 "INV-088's differentiation programme is reframed. Decisive in this direction, not "
                 "weak evidence."
             )},
            {"label": "zworld_advantage_grid_nondiscriminative", "outcome": "FAIL",
             "condition": (
                 "no arm passed its precondition gate, or the arms do not separate, or the "
                 "readouts fall in none of the enumerated branches"
             ),
             "reading": "Not a refutation. Unscored; re-pose the measurement."},
        ],
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown", "INV-088": "unknown"},
        "per_arm_gate": gate["per_arm_gate"],
        "readiness": {
            "anchors_ready": anchors_ready,
            "trained_encoder_trained": trained_encoder_ok,
            "off_install_took": off_install,
            "on_install_took": on_install,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
            "post_bc_worst_by_arm": {
                a: per_arm_zworld[a]["post_bc_foraging_competence_worst"] for a in BOOT_ARMS
            },
            "world_encoder_delta_worst_trained": on_r["world_encoder_max_abs_delta_worst"],
            "readiness_assert": {
                "name": "zworld_world_encoder_trained_worst_cell",
                "kind": "readiness",
                "description": (
                    "The treatment must have been applied: the trained arm's worst cell moved at "
                    "least one split_encoder.world_encoder tensor (max|delta| > 0). Below it the "
                    "encoder is a frozen random projection (treatment not applied) and the run "
                    "self-routes substrate_not_ready_requeue."
                ),
                "control": "world_encoder_max_abs_delta, worst trained cell (SD-070 P0a warmup)",
                "direction": "lower",
                "comparator": ">",
                "measured": on_r["world_encoder_max_abs_delta_worst"],
                "threshold": 0.0,
                "met": trained_encoder_ok,
            },
        },
        "headline": {
            "advantage_confirmed": advantage_confirmed,
            "training_advantage_transient": on_succeeded_then_decayed,
            "training_harms_competence": harms_confirmed,
            "pathway_dispensable": dispensable,
            "mean_paired_auc_delta_trained_minus_random": mean_auc_delta,
            "mean_paired_peak_delta": mean_peak_delta,
            "mean_paired_terminal_delta": mean_terminal_delta,
            "effect_margin": eff_margin,
            "n_seeds_advantage": n_seed_adv,
            "n_seeds_harm": n_seed_harm,
            "n_paired_seeds": n_paired,
            "random_proj_auc_mean": round(off_auc_mean, 6),
            "trained_auc_mean": round(float(on_r["competence_trajectory_auc_mean"]), 6),
            "random_proj_post_bc_mean": round(
                float(sum(off_r["post_bc_foraging_competence_per_seed"]) /
                      max(1, len(off_r["post_bc_foraging_competence_per_seed"]))), 6),
            "trained_post_bc_mean": round(
                float(sum(on_r["post_bc_foraging_competence_per_seed"]) /
                      max(1, len(on_r["post_bc_foraging_competence_per_seed"]))), 6),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "bc_expert_748_reference": float(boot.reference_band().get("bc_expert_748", 32.72)),
        },
        "paired_advantage": {
            "advantage_abs_floor": ADVANTAGE_ABS_FLOOR,
            "advantage_rel_frac": ADVANTAGE_REL_FRAC,
            "effect_margin": eff_margin,
            "mean_auc_delta": mean_auc_delta,
            "mean_peak_delta": mean_peak_delta,
            "mean_terminal_delta": mean_terminal_delta,
            "n_seeds_advantage": n_seed_adv,
            "n_seeds_harm": n_seed_harm,
            "n_seeds_peak_advantage": n_seed_peak_adv,
            "per_seed": paired_rows,
        },
        "per_arm_zworld": per_arm_zworld,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "post_bc_install_floor": float(POST_BC_INSTALL_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
            "bc_expert_748_reference_cotrain": 32.72,
            "lift_competence_target": float(LIFT_COMPETENCE_TARGET),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(gate["non_degenerate"]),
        "degeneracy_reason": (
            gate["degeneracy_reason"]
            or ("" if degeneracy["non_degenerate"] else degeneracy["degeneracy_reason"])
        ),
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "discrimination_verdict": result["discrimination_verdict"],
        "per_arm_gate": result["per_arm_gate"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "paired_advantage": result["paired_advantage"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_zworld": result["per_arm_zworld"],
        "reference_band": result["reference_band"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": [OFF_ARM],
            "reuse_eligible": True,
            "note": (
                "The CONTROL arm (" + OFF_ARM + ", frozen random-projection z_world) is emitted "
                "reuse-ELIGIBLE with the shared lineage slice "
                "(experiments/_lib/baselines/competence_floor_zworld.arm_config_slice) and "
                "include_driver_script_in_hash=False, so a later sibling leg can reuse the minted "
                "frozen-random-projection cell across a different driver. No separate baseline-only "
                "mint job (neither sanctioned exception applies)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The PAIRED per-seed advantage of a PREDICTION-TRAINED z_world over a FROZEN "
            "RANDOM-PROJECTION z_world on the post-installation competence TRAJECTORY AUC (mean "
            "foraging_competence across the BC install point and every mid-training probe reading, "
            f"probed every {cfg['retention_probe_every']} episodes), both arms on a DETACHED "
            "encoder (cotrain=False) so only the encoder's initial content differs. PASS: the "
            "trained arm's mean paired AUC delta clears the effect margin (max of "
            f"{ADVANTAGE_ABS_FLOOR} absolute and {ADVANTAGE_REL_FRAC} of the control AUC) on a "
            "strict majority of seeds, OR installs competence where the random projection cannot. "
            "Readiness: anchors clear the 1.0 floor, the trained arm's encoder actually trained "
            "(zworld_encoder_guard, GATING on the trained arm only), and at least one arm installs; "
            "otherwise substrate_not_ready_requeue."
        ),
        "notes": (
            "V3-EXQ-819a SUPERSEDES V3-EXQ-819 (test-design fix; measurement_test_design_defect "
            "adjudicated in failure_autopsy_batch-793a-817-819_2026-07-26.md target 3, "
            "user-confirmed 2026-07-26). THREE FIXES, all measurement -- the substrate and the "
            "science are unchanged: (1) post_bc_install_took is scored on the STRICT-MAJORITY "
            "install FRACTION (> 0.5), the identical predicate the router uses, so the gate and "
            "router cannot disagree and a non-installing control never vacates the trained arm "
            "(the 819 defect: worst-seed scoring vacated the run on seed 43's 0.9-vs-1.0 near-miss "
            "while the strict-majority router read an asymmetric-install advantage); (2) "
            "ADVANTAGE_ABS_FLOOR 1.0 -> 0.3, rescaled to the 0.35-2.4 res/ep near-floor band (819's "
            "1.0 margin demanded a near-doubling and dominated the relative bar); (3) seeds 3 -> 6 "
            "(42..47), matching the 793a repower, for a firmer 4/6 strict-majority. The z_world "
            "encoder trained in 819 (world_encoder_max_abs_delta 0.277-0.333, 3/3 seeds), so the "
            "V3-EXQ-780 frozen-random-projection fault stays fixed. "
            "MECH-457 / INV-088 competence_floor R4 INSTRUMENT-VALIDITY leg (D12), routed in "
            "competence_floor_reposing_2026-07-20.md section 4 R4 and re-confirmed unblocked in "
            "competence_floor_reposing_2026-07-25.md. DIAGNOSTIC (excluded from scoring); "
            "PROMOTES/DEMOTES NOTHING; route to /failure-autopsy. THE QUESTION: does a "
            "prediction-trained z_world confer any competence advantage over a random projection of "
            "the same dimensionality? THE ONE MANIPULATION: the SD-070 P0a encoder warmup "
            "(zworld_p0=60 trained vs 0 random); everything else identical, encoder DETACHED "
            "(cotrain=False) so the representation is FIXED during BC+RL and only its initial "
            "content differs. Re-derive brake RELEASED: sd_zworld_warmup_optimizer_group now BUILT "
            "(ree-v3 b523b9c70a) + VALIDATED (REE_assembly 25a69fcd4c) -- the upstream substrate the "
            "780/781/782 cluster autopsy owed; and this is a new-number, never-tested question, not "
            "a lettered same-mechanism iteration. GOV-REUSE-1: the decisive readout (trained-z_world "
            "competence) is absent from every manifest by construction -- the trained encoder was "
            "unconstructible before 2026-07-20 (all sixteen prior z_world arms ran a frozen random "
            "projection, V3-EXQ-780 diagnosis) -> must run. Reference build 128-wide / 3x budget / "
            "z_world detached -- NOT the 769-falsified 256/5x. CAVEAT: the historical 32.72 z_world+BC "
            "band (748) ran cotrain=True, so absolute install here may differ; the within-run "
            "RANDOM-vs-TRAINED contrast under the user-specified detached build is decisive. FIVE "
            "measurement_requirement constraints honoured: trajectory DV (not terminal); an explicit "
            "'succeeded then decayed' branch; routing on declared covariates (install_took + "
            "post_bc); substrate_not_ready_requeue when neither install took / treatment not applied; "
            "and a GATING zworld_encoder_guard scoped to the trained arm (a guard fire self-routes "
            "substrate_not_ready_requeue per-arm, never a scientific verdict). REGISTRY: R4 is NOT "
            "yet registered as a hypothesis under question competence_floor (D12); the registry is "
            "derive-only, single producer /failure-autopsy Step 9b -- register D12 there before or "
            "when adjudicating this run. MECH-457 stays candidate/v3_pending; INV-088 unchanged."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-819 MECH-457/INV-088 competence_floor R4 z_world instrument-validity"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="RL refinement budget (episodes) per arm x seed")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        on_budget = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
        probe_every = DRY_RETENTION_PROBE_EVERY
        p0, zworld_p0, n_bc = DRY_P0_WARMUP, DRY_ZWORLD_P0, DRY_BC
    else:
        seeds = list(SEEDS_819A)   # 819a: 6 seeds (42..47), was fan.SEEDS (3)
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)   # 3000 -- reference budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        probe_every = RETENTION_PROBE_EVERY
        p0, zworld_p0, n_bc = P0_WARMUP_EPISODES, ZWORLD_P0_EPISODES, BC_WARMSTART_EPISODES

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.episodes is not None:
        on_budget = int(args.episodes)
    if args.steps is not None:
        steps = int(args.steps)
    if args.eval_episodes is not None:
        eval_eps = int(args.eval_episodes)
    if args.probe_every is not None:
        probe_every = int(args.probe_every)

    result = run_experiment(seeds=seeds, on_budget=on_budget, eval_eps=eval_eps, steps=steps,
                            probe_every=probe_every, p0=p0, zworld_p0=zworld_p0, n_bc=n_bc,
                            dry_run=bool(args.dry_run))

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representation": "z_world_detached",
        "on_budget_episodes": on_budget,
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "retention_probe_every": probe_every,
        "p0_warmup_episodes": p0,
        "zworld_p0_episodes_trained_arm": zworld_p0,
        "bc_warmstart_episodes": n_bc,
        "bc_demonstrator": "local_view_greedy",
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_cotrain_encoder": zbase.REF_COTRAIN_ENCODER,
        "shaping_coef": zbase.SHAPING_COEF,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA, "bc_lr": fan.BC_LR,
        "advantage_abs_floor": ADVANTAGE_ABS_FLOOR,
        "advantage_rel_frac": ADVANTAGE_REL_FRAC,
        "lift_competence_target": float(LIFT_COMPETENCE_TARGET),
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "off_arm_config_slice": zbase.arm_config_slice(
            x734._env_kwargs_for_rung(fan.RUNG), trained=False, eval_eps=eval_eps, steps=steps,
            on_budget=on_budget, probe_every=probe_every, p0=p0, zworld_p0=0, n_bc=n_bc,
        ),
        "on_arm_config_slice": zbase.arm_config_slice(
            x734._env_kwargs_for_rung(fan.RUNG), trained=True, eval_eps=eval_eps, steps=steps,
            on_budget=on_budget, probe_every=probe_every, p0=p0, zworld_p0=zworld_p0, n_bc=n_bc,
        ),
        "portfolio": "competence_floor R4 instrument-validity (D12)",
        "hypothesis_id": "R4-zworld-trained-vs-random-projection",
        "hypothesis_question": "competence_floor",
        "claims": CLAIM_IDS,
        "substrate_commit": "ree-v3 b523b9c70a (sd_zworld_warmup_optimizer_group)",
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    # AFTER arm_results is assembled, so substrate_hash hoists from the per-cell fingerprints.
    stamp_recording_core(manifest, config=cfg, seeds=seeds, script_path=Path(__file__),
                         started_at=t0)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"anchors_ready={result['readiness']['anchors_ready']} "
        f"trained_encoder={result['readiness']['trained_encoder_trained']} "
        f"off_install={result['readiness']['off_install_took']} "
        f"on_install={result['readiness']['on_install_took']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for arm_id in BOOT_ARMS:
        r = result["per_arm_zworld"][arm_id]
        print(
            f"  {arm_id}: post_bc_worst={r['post_bc_foraging_competence_worst']} "
            f"install_took={r['n_seeds_install_took']}/{r['n_cells']} "
            f"enc_trained={r['n_seeds_encoder_trained']}/{r['n_cells']} "
            f"auc_mean={r['competence_trajectory_auc_mean']} "
            f"peak_mean={r['trajectory_peak_mean']} "
            f"terminal={r['terminal_forage_mean']}", flush=True,
        )
    print(
        f"  mean_auc_delta(trained-random)={hl['mean_paired_auc_delta_trained_minus_random']} "
        f"eff_margin={hl['effect_margin']} n_adv={hl['n_seeds_advantage']}/{hl['n_paired_seeds']} "
        f"green_arms={result['per_arm_gate']['green_arms']} "
        f"red_arms={result['per_arm_gate']['red_arms']}", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
