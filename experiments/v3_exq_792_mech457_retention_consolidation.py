#!/opt/local/bin/python3
"""V3-EXQ-792 -- MECH-457 GOV-FANOUT-1 RETENTION leg, H-retention-consolidation: does an UPDATE
CONSTRAINT (a trust-region KL anchor to the INSTALLED policy) let a BC-installed competent policy
survive RL refinement rather than being eroded?

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PRE-REGISTERED as hypothesis H-retention-consolidation under question qid "competence_floor" in
REE_assembly/evidence/planning/hypothesis_space_registry.v1.json.

THE HYPOTHESIS. There is no consolidation / protection pathway for an acquired policy: nothing
in the update rule preserves what BC installed, so refinement overwrites it. V3-EXQ-780 measured
the erosion directly on raw_view -- post-BC foraging competence 20.933 (3/3 seeds took the
install) eroded to 11.667 under unconstrained RL refinement. This leg changes the UPDATE
CONSTRAINT ONLY.

DESIGN. BC-install the raw_view policy to the ~20.9 competence point, then run the SAME reference
RL refinement under an unconstrained update vs a KL anchor to the INSTALLED policy, swept over
three anchor weights:
  * retcons_unconstrained (OFF / control) -- no anchor. THE SHARED LINEAGE BASELINE.
  * retcons_kl0p03        (ON) -- kl_anchor_coef 0.03
  * retcons_kl0p10        (ON) -- kl_anchor_coef 0.10
  * retcons_kl0p30        (ON) -- kl_anchor_coef 0.30

WHY A SWEEP RATHER THAN ONE GUESS -- AND WHY THREE POINTS, NOT TWO. Two independent reasons, the
second of which is a hard measurement constraint discovered in the substrate:

  (1) COVERAGE. An anchor too weak is indistinguishable from OFF; one too strong freezes the
      policy. A single guess that lands in either failure mode returns a confident null about a
      constraint that was never actually exercised at a readable strength.

  (2) THE ONLY AVAILABLE PROOF THAT THE ANCHOR BOUND IS THE CROSS-COEF DOSE-RESPONSE. The guard
      key mean_policy_kl_to_anchor_recent is emitted unconditionally, but on the OFF arm it is a
      HARD-CODED 0.0 SENTINEL, not a measured drift: mech457_explorer_classes.py:1036-1038 reads
      `kl_anchor.mean_kl_recent() if kl_anchor is not None else 0.0`, and the OFF arm constructs
      no PolicyKLAnchor, so no KL is ever recorded there. The intuitive check -- "the anchored
      arm's realised KL should be lower than the control's" -- therefore compares a measurement
      against a sentinel and reads BACKWARDS (every anchored arm looks like it drifted MORE than
      an unconstrained control). The valid check is WITHIN the anchored arms, where all three
      measure the same quantity against their own snapshot: realised KL must DECREASE
      MONOTONICALLY as kl_anchor_coef increases. That needs >= 3 points to be a dose-response
      rather than a single ordering, which is why the sweep is three-valued.

COEFFICIENT SCALE, JUSTIFIED. The KL penalty is added to the SAME policy loss as the entropy
bonus, and the entropy bonus is the only other distribution-level regulariser on the same logits,
so it is the natural scale reference. The reference build anneals entropy_beta 0.10 -> 0.03
(ON_ENTROPY_BETA_START / ON_ENTROPY_BETA_END). The sweep brackets that band and extends one step
beyond it:
  * 0.03 = the TERMINAL entropy beta -- the weakest regulariser the design already tolerates.
           Guards the "too weak to distinguish from OFF" end.
  * 0.10 = the INITIAL entropy beta -- the strongest distribution-level regulariser already in
           the loss. The a-priori most likely to bind without dominating.
  * 0.30 = 3x the strongest existing regulariser. Guards the "too strong, policy frozen" end and
           gives the dose-response its third, widest-spaced point.

RETENTION BY FREEZING IS NOT RETENTION BY CONSOLIDATION -- THE GRID BRANCH THIS LEG NEEDS AND
V3-EXQ-788'S DID NOT. The KL anchor, unlike the critic swap, has a TRIVIALLY-RETAINING degenerate
limit: as coef grows the policy stops moving, competence stays pinned at the installed value, and
retained_fraction -> 1.0. Scored on retention alone that is a PASS, and it would be a vacuous one.
So the grid enumerates retention_by_policy_freezing as its own branch, and the PASS branch
additionally requires that the arm RETAINED PLASTICITY (realised KL above a floor) and IMPROVED
above the install at its trajectory peak. Freezing is NOT scored as a null either -- "competence
is preservable, but only by abolishing plasticity" is a real retention/plasticity TRADEOFF
finding, and collapsing it into either PASS or the null would misreport it.

ANTI-ALIAS (load-bearing, and structural rather than conventional).
  vs H-retention-auxiliary-decay (V3-EXQ-789): the anchor is a deep copy of the LEARNER'S OWN
    weights at the post-install checkpoint. PolicyKLAnchor never sees bc_demo and works with
    bc_demo=None. Anchoring via bc_aux_coef would anchor to the DEMONSTRATOR and alias.
  vs H-retention-critic (V3-EXQ-788): the KL term is a function of `logits` alone and puts
    EXACTLY ZERO gradient on value_head / value_bins; fan.critic_value_loss is byte-identical on
    both branches. HONEST LIMIT, carried from the substrate's own docstring: the trunk is shared,
    so the critic's INPUT FEATURES do move -- the exact mirror of 788's situation, whose contract
    C2 asserted only that the CE loss puts no gradient on the policy HEAD.
  use_distributional_critic is False and bc_aux_coef is the constant baseline 0.5 on EVERY arm
  here, so neither sibling locus moves.

DV-SYMMETRY DECLARATION (mandatory, per arm). The DV is the foraging-competence trajectory, read
through the deterministic-argmax eval policy, so its symmetry group is the one that fixes an
argmax: uniform additive shifts across candidates, and monotone rescalings.
  * retcons_unconstrained -- no manipulation; it IS the reference.
  * retcons_kl0p03 / kl0p10 / kl0p30 -- the manipulation is a gradient term
    coef * KL(pi || pi_ref) whose gradient pushes each candidate's live logit toward that
    candidate's OWN frozen reference logit. Because the reference logits differ ACROSS
    candidates, the induced logit change is per-candidate differential, NOT a broadcast scalar,
    and it alters the learned weights rather than adding a constant at read time. It is
    therefore NOT invariant under argmax or under monotone rescaling, and the measured delta is
    not an arithmetic identity. The one invariant limit -- coef so large the weights never move,
    making the arm identical to the installed policy -- is exactly the frozen case, and it is
    routed to its own branch rather than being read as retention.

THE DV IS A TRAJECTORY, NOT A TERMINAL SCALAR. cfg.retention_probe_every wires the substrate's
non-perturbing mid-training competence probe (train_a2c snapshots and restores the torch/numpy/
random streams around every reading, so measurement neutrality is a substrate guarantee). At
250-episode cadence over the 3000-episode budget that is 12 readings per (arm x seed), each
recorded in full. Terminal-only measurement is what kept this deficit invisible for ten legs.

INTERPRETATION GRID (six branches; routing consumes the DECLARED COVARIATES and the trajectory
SHAPE before the terminal criterion, which is the defect that sank V3-EXQ-780):
  * substrate_not_ready_requeue                   -- the install did NOT take (or an anchor is
      sub-floor). UNINFORMATIVE about retention. NEVER a retention verdict, and never
      substrate_ceiling / substrate_conditional / does_not_support.
  * retention_consolidation_protects_competence   -- PASS. Some anchored arm holds the installed
      competence, beats the unconstrained arm by the declared margin, AND kept its plasticity.
  * retention_by_policy_freezing                  -- an anchored arm "retains" only by not
      moving (realised KL below the plasticity floor and no improvement above the install). A
      retention/plasticity TRADEOFF finding, not a PASS and not the null.
  * retention_consolidation_succeeded_then_decayed-- an anchored arm ROSE to at/above the
      lift-competence target at its trajectory PEAK and then FELL below the retention floor.
  * retention_eroded_under_all_anchors            -- THE DECLARED NULL. The installed prior
      erodes at every anchor weight -> retention is NOT a drift-protection problem.
  * retention_grid_nondiscriminative              -- no arm passed its gate, or the anchor never
      bound (no dose-response in realised KL). Not a refutation.

MULTI-ARM GATE. experiments/_lib/precondition_gate.py -- per-arm gates aggregated with
aggregate_arm_gates so non_degenerate = ANY arm green. One arm's unmet precondition must NEVER
vacate another's valid result (the V3-EXQ-785 defect). The anchor-installed precondition is
scoped OUT of the unconstrained control via applies_to: the control is DEFINED by the absence of
an anchor, so asserting it there would make its gate structurally un-passable and collapse the
four-arm design to three.

MINT / REUSE. The OFF arm is built from the shared lineage module
(baselines.off_path_config_slice) and emitted reuse-ELIGIBLE with
include_driver_script_in_hash=False, so a sibling leg's different driver can reuse it. NO REUSE
IS ATTEMPTED FROM V3-EXQ-788's MINT, deliberately and on a substrate ground rather than an
oversight: 788 ran at substrate_hash 953cb8d0... on 2026-07-19T23:55Z, and the enabling commit
ree-v3 399b17caed (2026-07-20) edited THREE files under experiments/_lib/** --
mech457_explorer_classes.py, mech457_bootstrap_explorer.py and baselines/mech457_retention.py --
all of which are inside the substrate glob. The fingerprint therefore CORRECTLY refuses 788's
cell. That refusal is a false MISS, which is the safe direction of the governing asymmetry (a
false HIT would corrupt a conclusion; a false MISS only costs compute). This leg mints the OFF
cell fresh at the post-399b17caed substrate_hash, which V3-EXQ-789 can then reuse if it runs on
the same substrate.

evidence_direction = "unknown" (DIAGNOSTIC; verdict lives in interpretation.label /
discrimination_verdict, adjudicated by /failure-autopsy).

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
import experiments._lib.baselines.mech457_retention as baselines  # noqa: E402
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_792_mech457_retention_consolidation"
QUEUE_ID = "V3-EXQ-792"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# --- Probe cadence (the measurement constraint the whole leg exists to satisfy) ------------
# 3000-episode reference budget / 250 = 12 readings per (arm x seed). Declared as a module
# constant because it is fingerprint-declared in the config_slice: a probed and an unprobed cell
# are bit-identical COMPUTATIONS but are not interchangeable ARTIFACTS.
RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# --- The declared manipulation: anchor weights -------------------------------------------
# Scale-justified against the entropy bonus, the only other distribution-level regulariser on the
# same logits in the same loss (ON_ENTROPY_BETA_START 0.10 -> ON_ENTROPY_BETA_END 0.03). See the
# module docstring for the full argument.
KL_ANCHOR_COEFS: Tuple[float, ...] = (0.03, 0.10, 0.30)

# --- Pre-registered retention thresholds (declared; never derived from the run) ------------
# retained_fraction = terminal trajectory competence / installed (post-BC) competence.
RETAINED_FRACTION_FLOOR = 0.5          # >= half the installed competence survives refinement
RETENTION_ARM_MARGIN = 0.15            # an anchored arm must beat OFF by this in retained frac
PEAK_SUCCESS_TARGET = baselines.LIFT_COMPETENCE_TARGET   # ~13.05 res/ep
MIN_TRAJECTORY_READINGS = 1.5          # need >= 2 readings for a SHAPE (floor, strict >)

# Anti-freeze. Realised mean KL to the frozen snapshot, in nats. Below this the policy never
# meaningfully left the installed weights, so any "retention" is FREEZING, not consolidation.
KL_PLASTICITY_FLOOR = 1e-3
# An arm has "improved above the install" when its trajectory peak clears the installed
# competence by this relative margin. Freezing shows retention WITHOUT improvement.
PEAK_IMPROVEMENT_MARGIN = 0.05
# The anchor must BIND: realised KL must fall as the coefficient rises. Measured WITHIN the
# anchored arms only -- the OFF arm's mean_policy_kl_to_anchor_recent is a 0.0 sentinel, not a
# measured drift (see the module docstring).
KL_DOSE_RESPONSE_MIN_SPREAD = 1e-4     # weakest-minus-strongest realised KL must exceed this

# The install / reference build. Re-exported from the shared lineage module so this driver
# cannot drift from it.
REF_REPRESENTATION = baselines.REF_REPRESENTATION            # raw_view ONLY
REF_ACTOR_CRITIC_HIDDEN = baselines.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = baselines.REF_BUDGET_MULTIPLIER      # 3x
POST_BC_INSTALL_FLOOR = baselines.POST_BC_INSTALL_FLOOR      # 1.0 competence floor

OFF_KIND = "unconstrained"


def _coef_tag(coef: float) -> str:
    """ASCII-safe, filesystem-safe tag for a coefficient (0.03 -> 'kl0p03')."""
    return "kl" + ("%.2f" % float(coef)).replace(".", "p")


CFG_KINDS: Tuple[str, ...] = (OFF_KIND,) + tuple(_coef_tag(c) for c in KL_ANCHOR_COEFS)
COEF_BY_KIND: Dict[str, float] = {OFF_KIND: 0.0}
COEF_BY_KIND.update({_coef_tag(c): float(c) for c in KL_ANCHOR_COEFS})


def _arm_id(cfg_kind: str) -> str:
    return f"retcons_{cfg_kind}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(k) for k in CFG_KINDS)
OFF_ARM = _arm_id(OFF_KIND)
ANCHORED_ARMS: Tuple[str, ...] = tuple(_arm_id(k) for k in CFG_KINDS if k != OFF_KIND)
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


# ---------------------------------------------------------------------------------------
# Precondition specs (multi-arm gate). Every entry carries a numeric measured + threshold so
# build_experiment_indexes.py can recompute `met`; all are FLOORS (direction defaults to
# "lower"), so none needs the "upper" ceiling tag.
# ---------------------------------------------------------------------------------------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="local_view_greedy_clears_floor_at_d3",
        description=(
            "LocalViewGreedyPolicy reading the SAME 5x5 resource_field_view forages above the "
            "1.0 competence floor at D3 -- the positive control that the env is solvable from "
            "the local view. Below-floor means the substrate/env is not ready, NOT that the "
            "update constraint failed."
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
            "THE LOAD-BEARING READINESS PRECONDITION. The BC install must have TAKEN before RL: "
            "post_bc_foraging_competence (measured pre-RL, post warm-start) clears the 1.0 "
            "floor on the WORST seed of this arm. 780 measured 20.933 here on raw_view with "
            "3/3 seeds taking. An install that did not take is UNINFORMATIVE about retention: "
            "there is no installed prior to retain, so the run self-routes "
            "substrate_not_ready_requeue rather than any retention verdict."
        ),
        control="post-BC / pre-RL foraging_competence of this arm's worst seed vs the 1.0 floor",
        threshold=float(POST_BC_INSTALL_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="competence_trajectory_readings",
        description=(
            "The retention DV is a TRAJECTORY, so the arm's worst cell must carry at least two "
            "probe readings -- one reading has no shape and cannot separate 'retained' from "
            "'succeeded then decayed'."
        ),
        control="number of mid-training competence probe readings in this arm's worst cell",
        threshold=float(MIN_TRAJECTORY_READINGS),
        kind="measurability",
        # Design-time proof: the cadence and the budget are both pre-registered, so an
        # unsatisfiable probe schedule is caught before any compute is spent.
        structural_max=lambda ctx: float(
            int(ctx["n_episodes"]) // max(1, int(ctx["probe_every"]))
        ),
    ),
    PreconditionSpec(
        name="policy_kl_anchor_installed",
        description=(
            "The anchored arm must actually be running the KL anchor (1 = installed). Scoped "
            "OUT of the unconstrained control, where the anchor is not merely absent but is the "
            "very thing being controlled against -- asserting it there would make the control's "
            "gate structurally un-passable and collapse the four-arm design (the V3-EXQ-785 "
            "regime-conditioning lesson)."
        ),
        control="guard['policy_kl_anchor_installed'] as realised by train_a2c for this arm",
        threshold=0.5,
        kind="manipulation_active",
        applies_to=lambda ctx: bool(ctx["use_policy_kl_anchor"]),
        applies_note=(
            "anchored arms only -- the unconstrained CONTROL is defined by the absence of the "
            "anchor, so this precondition is not meaningful for it"
        ),
    ),
    PreconditionSpec(
        name="policy_kl_penalty_was_exercised",
        description=(
            "The anchored arm must have actually EVALUATED the penalty: a realised "
            "mean_policy_kl_to_anchor_recent strictly above 0 proves PolicyKLAnchor.penalty() "
            "was called and recorded, rather than the anchor being constructed and never "
            "reached. This is a MEASURABILITY check, NOT the anti-freeze test: a near-zero but "
            "positive KL means the anchor bound very tightly, which is a substantive reading "
            "(retention_by_policy_freezing) and is routed by the grid, never gated here. Scoped "
            "out of the unconstrained control, whose guard value is a hard-coded 0.0 sentinel "
            "and not a measured drift."
        ),
        control="guard['mean_policy_kl_to_anchor_recent'] for this anchored arm, vs 0",
        threshold=0.0,
        kind="measurability",
        applies_to=lambda ctx: bool(ctx["use_policy_kl_anchor"]),
        applies_note=(
            "anchored arms only -- the control constructs no PolicyKLAnchor, so its guard "
            "value is a sentinel rather than a measurement and cannot be scored"
        ),
    ),
)


PRECONDITION_BY_NAME: Dict[str, PreconditionSpec] = {s.name: s for s in PRECONDITION_SPECS}

# --- Readiness-anchor reachability reference (V3-EXQ-778d lesson) --------------------------
# The install-took anchor self-routes to substrate_not_ready_requeue, so the gate must be
# provably REACHABLE by its known-positive control before the run spends compute -- a predicate
# narrower than the state it anchors to reports met=false forever and mislabels an
# instrument-specification gap as a substrate verdict. These are V3-EXQ-780's RECORDED per-seed
# raw_view post-BC competences (3/3 seeds took), scored by THE SHIPPED PREDICATE
# (PreconditionSpec.met_for for post_bc_install_took), not by a copy of it.
POST_BC_REFERENCE_CELLS_780: Tuple[float, ...] = (17.75, 26.15, 18.9)
POST_BC_REFERENCE_SOURCE = (
    "v3_exq_780_mech457_bc_prior_discrimination_20260718T123325Z_v3.json "
    "headline.per_representation.raw_view.treat_post_bc_forage_per_seed "
    "(mean 20.933333, 3/3 seeds took)"
)
POST_BC_ANCHOR_REACHABILITY_THRESHOLD = 1.0   # every reference cell must clear the install floor


def _make_cfg(cfg_kind: str, on_budget: int, probe_every: int):
    """The shared reference RL-refinement config, varying ONLY the update constraint.

    Everything else -- bc_aux_coef, actor_critic_hidden, budget, credit replay, the drive anneal,
    the install, and use_distributional_critic (False everywhere) -- comes from
    baselines.reference_config() and is therefore IDENTICAL across all four arms by construction.
    That is the leg's anti-alias: the value estimator and the auxiliary are untouched; only the
    UPDATE CONSTRAINT differs.

    train_a2c RAISES when use_policy_kl_anchor and a positive kl_anchor_coef disagree (either
    spelling alone yields an arm identical to the control while labelled as anchored), so the two
    are derived together from one source of truth here rather than passed independently.
    """
    if cfg_kind not in CFG_KINDS:
        raise ValueError(f"unknown cfg_kind {cfg_kind!r}")
    coef = float(COEF_BY_KIND[cfg_kind])
    return baselines.reference_config(
        on_budget,
        use_policy_kl_anchor=(coef > 0.0),
        kl_anchor_coef=coef,
        retention_probe_every=int(probe_every),
    )


def _config_slice(cfg_kind: str, env_kwargs: Dict[str, Any], eval_eps: int, steps: int,
                  on_budget: int, probe_every: int) -> Dict[str, Any]:
    """All arms declare the SHARED lineage slice; an anchored arm then overrides its own arm_id
    and folds in its (anchored) config. Anchoring every arm on off_path_config_slice makes the
    OFF-cell match with the sibling legs structural rather than accidental."""
    base = baselines.off_path_config_slice(
        env_kwargs, eval_eps=int(eval_eps), steps=int(steps),
        on_budget=int(on_budget), retention_probe_every=int(probe_every),
    )
    if cfg_kind == OFF_KIND:
        return base
    base["arm_id"] = _arm_id(cfg_kind)
    base["kind"] = "mech457_retention_consolidation_treatment"
    base.update(_make_cfg(cfg_kind, on_budget, probe_every).as_slice())
    return base


def _run_boot_cell(cfg_kind: str, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind)
    cfg = _make_cfg(cfg_kind, on_budget, probe_every)
    # Every arm -- control and anchored alike -- is constructed through the shared lineage
    # builder. Unlike the critic swap (a REP-CONSTRUCTION knob that build_off_arm deliberately
    # does not forward), the KL anchor is an UPDATE-RULE knob applied inside
    # train_bootstrap_explorer and reaches the trainer via cfg, so there is no treatment branch
    # to special-case here. See baselines.build_off_arm's docstring, which warns against
    # "symmetrising" the two.
    rep_agent = baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)

    # Phase 1 -- BC install to the ~20.9 competence point, then measure whether it TOOK.
    # This runs BEFORE train_a2c, which is why PolicyKLAnchor's "snapshot at entry" IS the
    # post-install checkpoint and needs no explicit checkpoint argument.
    install = baselines.install_bc_prior(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    post_bc = float(install["post_bc_foraging_competence"])

    # Phase 2 -- the SAME RL refinement under this arm's update constraint, probed on a cadence.
    probe_fn = baselines.make_probe_fn(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    guard = baselines.train_off_arm(
        rep_agent, seed, env_kwargs, steps=steps, arm_label=arm_id, cfg=cfg,
        demo=install["demo"], probe_fn=probe_fn,
    )
    trajectory: List[Dict[str, Any]] = list(guard.get("competence_trajectory", []))

    # Phase 3 -- unshaped terminal eval (same statistic as the anchors and the probe).
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)

    traj_vals = [float(r.get("foraging_competence", 0.0)) for r in trajectory]
    peak = round(max(traj_vals), 6) if traj_vals else 0.0
    terminal_traj = round(traj_vals[-1], 6) if traj_vals else 0.0
    retained = baselines.retained_fraction(trajectory, post_bc)
    half_life = baselines.competence_half_life(trajectory, post_bc)
    mean_kl = float(guard.get("mean_policy_kl_to_anchor_recent", 0.0))

    row["post_bc_foraging_competence"] = round(post_bc, 6)
    row["install_took"] = bool(install["install_took"])
    row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
    # The declared manipulation, as REALISED by the substrate (not as requested by the driver).
    row["use_policy_kl_anchor"] = bool(guard.get("policy_kl_anchor_installed", False))
    row["kl_anchor_coef"] = round(float(guard.get("policy_kl_anchor_coef", 0.0)), 6)
    row["mean_policy_kl_to_anchor_recent"] = round(mean_kl, 6)
    # FULL per-seed trajectory -- never collapsed to a terminal scalar.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    row["retained_fraction"] = retained
    row["competence_half_life_episodes"] = half_life
    row["peak_cleared_success_target"] = bool(peak >= PEAK_SUCCESS_TARGET)
    # Anti-freeze readouts, per cell. "Improved" separates consolidation from freezing.
    row["peak_improved_above_install"] = bool(
        post_bc > 0.0 and peak >= post_bc * (1.0 + PEAK_IMPROVEMENT_MARGIN)
    )
    row["retained_plasticity"] = bool(mean_kl > KL_PLASTICITY_FLOOR)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["mean_bc_aux_action_match_recent"] = float(guard.get("mean_bc_aux_action_match_recent", 0.0))
    row["bc_aux_coef_first"] = guard.get("bc_aux_coef_first")
    row["bc_aux_coef_last"] = guard.get("bc_aux_coef_last")
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


def _arm_contexts(on_budget: int, probe_every: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": _arm_id(k),
            "cfg_kind": k,
            "use_policy_kl_anchor": (COEF_BY_KIND[k] > 0.0),
            "kl_anchor_coef": float(COEF_BY_KIND[k]),
            "n_episodes": int(on_budget),
            "probe_every": int(probe_every),
        }
        for k in CFG_KINDS
    ]


def run_experiment(seeds: List[int], on_budget: int, eval_eps: int, steps: int,
                   probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-retention-consolidation "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep={REF_REPRESENTATION}, ON_budget={on_budget}, probe_every={probe_every}, "
        f"eval={eval_eps}, steps={steps}; ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, "
        f"budget_mult={REF_BUDGET_MULTIPLIER}; "
        f"manipulation=policy KL anchor coefs {list(KL_ANCHOR_COEFS)} ONLY)",
        flush=True,
    )
    arm_ctxs = _arm_contexts(on_budget, probe_every)
    # Design-audit BEFORE any compute: refuse a run carrying a structurally unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)
    # ... and refuse a run whose install-took anchor its own known-positive control cannot clear.
    anchor_reachability = assert_anchor_reachable(
        anchor_name="post_bc_install_took",
        reference_cells=list(POST_BC_REFERENCE_CELLS_780),
        score_fn=PRECONDITION_BY_NAME["post_bc_install_took"].met_for,
        threshold=POST_BC_ANCHOR_REACHABILITY_THRESHOLD,
        reference_source=POST_BC_REFERENCE_SOURCE,
    )
    print(
        f"anchor reachability: {anchor_reachability['anchor_name']} "
        f"reference={anchor_reachability['n_reference_scored_true']}/"
        f"{anchor_reachability['n_reference_cells']} "
        f"reachable={anchor_reachability['reachable']}", flush=True,
    )

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        is_boot = arm_id in BOOT_ARMS
        is_off = (arm_id == OFF_ARM)
        if is_boot:
            slice_cfg = _config_slice(cfg_kind, env_kwargs, eval_eps, steps, on_budget,
                                      probe_every)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID,
                         "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        # OFF arm mints cross-driver-reusable (no driver script in the hash) so a sibling leg can
        # reuse it; the anchored arms are leg-specific and keep the driver in their hash.
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=not (is_off or not is_boot)) as cell:
            if is_boot:
                row = _run_boot_cell(cfg_kind, env_kwargs, seed, on_budget, eval_eps, steps,
                                     probe_every)
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

    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed, None)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    anchors_ready = bool(
        local_view_mean > COMPETENCE_RESOURCE_FLOOR and oracle_mean > COMPETENCE_RESOURCE_FLOOR
    )

    if anchors_ready:
        for cfg_kind in CFG_KINDS:
            for seed in seeds:
                _run_cell(_arm_id(cfg_kind), seed, cfg_kind)
    else:
        print(
            f"anchors UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping boot training -> substrate_not_ready_requeue", flush=True,
        )

    # ---- per-arm retention readouts (trajectory-shaped, worst-cell reported) --------------
    def _cells_for(arm_id: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == arm_id]

    def _worst(cells: List[Dict[str, Any]], key: str, default: float) -> Tuple[float, Any]:
        if not cells:
            return float(default), None
        best = min(cells, key=lambda c: float(c.get(key, default)))
        return float(best.get(key, default)), best.get("seed")

    per_arm_retention: Dict[str, Any] = {}
    for cfg_kind in CFG_KINDS:
        arm_id = _arm_id(cfg_kind)
        cells = _cells_for(arm_id)
        post_bc_worst, post_bc_worst_seed = _worst(cells, "post_bc_foraging_competence", 0.0)
        n_read_worst, n_read_worst_seed = _worst(cells, "n_trajectory_readings", 0.0)
        kl_worst, kl_worst_seed = _worst(cells, "mean_policy_kl_to_anchor_recent", 0.0)
        retained_vals = [c.get("retained_fraction") for c in cells]
        retained_num = [float(v) for v in retained_vals if v is not None]
        retained_mean = round(float(sum(retained_num) / len(retained_num)), 6) if retained_num else None
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        kls = [float(c.get("mean_policy_kl_to_anchor_recent", 0.0)) for c in cells]
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        n_peak_cleared = int(sum(1 for c in cells if bool(c.get("peak_cleared_success_target", False))))
        n_improved = int(sum(1 for c in cells if bool(c.get("peak_improved_above_install", False))))
        n_plastic = int(sum(1 for c in cells if bool(c.get("retained_plasticity", False))))
        n_retained = int(sum(1 for v in retained_num if v >= RETAINED_FRACTION_FLOOR))
        n_cells = len(cells)
        per_arm_retention[arm_id] = {
            "arm_id": arm_id,
            "use_policy_kl_anchor": bool(COEF_BY_KIND[cfg_kind] > 0.0),
            "kl_anchor_coef": float(COEF_BY_KIND[cfg_kind]),
            "n_cells": n_cells,
            "post_bc_foraging_competence_per_seed": [
                float(c.get("post_bc_foraging_competence", 0.0)) for c in cells
            ],
            "post_bc_foraging_competence_worst": round(post_bc_worst, 6),
            "post_bc_worst_seed": post_bc_worst_seed,
            "n_seeds_install_took": n_install_took,
            "install_took_strict_majority": bool(n_cells and n_install_took > (n_cells / 2.0)),
            "n_trajectory_readings_worst": int(n_read_worst),
            "n_trajectory_readings_worst_seed": n_read_worst_seed,
            "retained_fraction_per_seed": retained_vals,
            "retained_fraction_mean": retained_mean,
            "n_seeds_retained": n_retained,
            "retained_strict_majority": bool(n_cells and n_retained > (n_cells / 2.0)),
            "trajectory_peak_per_seed": [round(p, 6) for p in peaks],
            "trajectory_peak_mean": round(float(sum(peaks) / len(peaks)), 6) if peaks else 0.0,
            "n_seeds_peak_cleared_success_target": n_peak_cleared,
            "peak_cleared_strict_majority": bool(n_cells and n_peak_cleared > (n_cells / 2.0)),
            # Anti-freeze block. Realised KL is reported per seed AND worst-cell, never only as a
            # mean, so a single frozen seed cannot hide behind an in-band average.
            "mean_policy_kl_to_anchor_per_seed": [round(k, 6) for k in kls],
            "mean_policy_kl_to_anchor_mean": round(float(sum(kls) / len(kls)), 6) if kls else 0.0,
            "mean_policy_kl_to_anchor_worst": round(kl_worst, 6),
            "mean_policy_kl_to_anchor_worst_seed": kl_worst_seed,
            "n_seeds_retained_plasticity": n_plastic,
            "plasticity_strict_majority": bool(n_cells and n_plastic > (n_cells / 2.0)),
            "n_seeds_peak_improved_above_install": n_improved,
            "improved_strict_majority": bool(n_cells and n_improved > (n_cells / 2.0)),
            "competence_half_life_episodes_per_seed": [
                c.get("competence_half_life_episodes") for c in cells
            ],
            "terminal_forage_per_seed": [round(v, 6) for v in per_arm_forage[arm_id]],
            "terminal_forage_mean": round(_mean(arm_id), 6),
        }

    # ---- multi-arm gate (per-arm; a red arm NEVER vacates a green one) --------------------
    arm_gates = []
    for ctx in arm_ctxs:
        arm_id = ctx["id"]
        r = per_arm_retention[arm_id]
        measured = {
            "local_view_greedy_clears_floor_at_d3": round(local_view_mean, 6),
            "greedy_oracle_clears_floor_at_d3": round(oracle_mean, 6),
            "post_bc_install_took": float(r["post_bc_foraging_competence_worst"]),
            "competence_trajectory_readings": float(r["n_trajectory_readings_worst"]),
        }
        if ctx["use_policy_kl_anchor"]:
            cells = _cells_for(arm_id)
            installed_all = bool(cells) and all(
                bool(c.get("use_policy_kl_anchor", False)) for c in cells
            )
            measured["policy_kl_anchor_installed"] = 1.0 if installed_all else 0.0
            measured["policy_kl_penalty_was_exercised"] = float(
                r["mean_policy_kl_to_anchor_worst"]
            )
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, PRECONDITION_SPECS, measured=measured))
    gate = aggregate_arm_gates(arm_gates)

    # ---- anchor dose-response: the ONLY valid proof the anchor BOUND ----------------------
    # Measured WITHIN the anchored arms. The OFF arm's mean_policy_kl_to_anchor_recent is a
    # hard-coded 0.0 sentinel (no PolicyKLAnchor is constructed there), NOT a measured drift, so
    # comparing an anchored arm's KL against it would read backwards.
    anchored_kl_by_coef = [
        (float(per_arm_retention[a]["kl_anchor_coef"]),
         per_arm_retention[a]["mean_policy_kl_to_anchor_mean"])
        for a in ANCHORED_ARMS
    ]
    anchored_kl_by_coef.sort(key=lambda t: t[0])
    kl_series = [float(v) for _c, v in anchored_kl_by_coef]
    kl_monotone_decreasing = bool(
        len(kl_series) >= 2
        and all(kl_series[i] >= kl_series[i + 1] for i in range(len(kl_series) - 1))
    )
    kl_spread = round(float(max(kl_series) - min(kl_series)), 8) if kl_series else 0.0
    anchor_bound = bool(kl_monotone_decreasing and kl_spread > KL_DOSE_RESPONSE_MIN_SPREAD)

    # ---- routing: covariates + trajectory SHAPE + plasticity first, terminal criterion last --
    off_r = per_arm_retention[OFF_ARM]
    off_retained = bool(off_r["retained_strict_majority"])
    off_frac = off_r["retained_fraction_mean"]

    def _margin_over_off(arm_id: str) -> Optional[float]:
        on_frac = per_arm_retention[arm_id]["retained_fraction_mean"]
        if on_frac is None or off_frac is None:
            return None
        return round(float(on_frac) - float(off_frac), 6)

    install_took_all = bool(
        all(per_arm_retention[a]["install_took_strict_majority"] for a in BOOT_ARMS)
    )

    # An anchored arm CONSOLIDATES when it retains, beats the control by the declared margin, and
    # kept its plasticity (it moved, and it improved above the install). An arm that retains
    # WITHOUT plasticity is FROZEN -- retention purchased by abolishing learning, which is a
    # tradeoff finding rather than a consolidation success or a null.
    per_arm_verdict: Dict[str, Any] = {}
    for arm_id in ANCHORED_ARMS:
        r = per_arm_retention[arm_id]
        margin = _margin_over_off(arm_id)
        retains = bool(r["retained_strict_majority"])
        beats = bool(margin is not None and margin >= RETENTION_ARM_MARGIN)
        plastic = bool(r["plasticity_strict_majority"])
        improved = bool(r["improved_strict_majority"])
        per_arm_verdict[arm_id] = {
            "arm_id": arm_id,
            "kl_anchor_coef": float(r["kl_anchor_coef"]),
            "retained_fraction_mean": r["retained_fraction_mean"],
            "retained_fraction_margin_over_unconstrained": margin,
            "retains_strict_majority": retains,
            "beats_unconstrained_by_margin": beats,
            "retained_plasticity_strict_majority": plastic,
            "peak_improved_above_install_strict_majority": improved,
            "consolidates": bool(retains and beats and plastic and improved),
            "frozen": bool(retains and not plastic and not improved),
            "succeeded_then_decayed": bool(r["peak_cleared_strict_majority"] and not retains),
        }

    consolidating_arms = [a for a in ANCHORED_ARMS if per_arm_verdict[a]["consolidates"]]
    frozen_arms = [a for a in ANCHORED_ARMS if per_arm_verdict[a]["frozen"]]
    decayed_arms = [a for a in ANCHORED_ARMS if per_arm_verdict[a]["succeeded_then_decayed"]]
    eroded_under_all = bool(
        not consolidating_arms and not frozen_arms and not decayed_arms
        and not any(per_arm_verdict[a]["retains_strict_majority"] for a in ANCHORED_ARMS)
    )
    c_load_bearing = bool(consolidating_arms)

    if not anchors_ready or not install_took_all:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not gate["non_degenerate"]:
        outcome, label = "FAIL", "retention_grid_nondiscriminative"
    elif not anchor_bound:
        outcome, label = "FAIL", "retention_grid_nondiscriminative"
    elif c_load_bearing:
        outcome, label = "PASS", "retention_consolidation_protects_competence"
    elif frozen_arms:
        outcome, label = "FAIL", "retention_by_policy_freezing"
    elif decayed_arms:
        outcome, label = "FAIL", "retention_consolidation_succeeded_then_decayed"
    elif eroded_under_all:
        outcome, label = "FAIL", "retention_eroded_under_all_anchors"
    else:
        outcome, label = "FAIL", "retention_grid_nondiscriminative"

    degeneracy = check_degeneracy({
        "d3_boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        },
        "anchored_realised_kl_across_coefs": {"values": kl_series},
    })

    criteria_by_arm: Dict[str, List[str]] = {
        OFF_ARM: ["C_unconstrained_control_erodes_installed_competence"],
    }
    for arm_id in ANCHORED_ARMS:
        criteria_by_arm[arm_id] = [f"C_{arm_id}_consolidates_installed_competence"]
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm, gate,
        extra={
            "C_unconstrained_control_erodes_installed_competence": bool(
                degeneracy["non_degenerate"]
            ),
            **{
                # An arm that "retained" only by freezing did NOT discriminate consolidation --
                # its criterion passed for a degenerate reason and must not be scored as one.
                f"C_{a}_consolidates_installed_competence": bool(
                    degeneracy["non_degenerate"] and not per_arm_verdict[a]["frozen"]
                )
                for a in ANCHORED_ARMS
            },
        },
    )
    criteria_nd["boot_arm_vs_anchor_foraging_spread"] = bool(degeneracy["non_degenerate"])
    criteria_nd["install_took_on_all_arms"] = install_took_all
    criteria_nd["anchor_bound_dose_response"] = anchor_bound
    criteria_nd["trajectory_has_shape_on_all_arms"] = bool(
        all(
            per_arm_retention[a]["n_trajectory_readings_worst"] > MIN_TRAJECTORY_READINGS
            for a in BOOT_ARMS
        )
    )

    criteria = [
        {"name": "C_anchored_arm_consolidates_installed_competence",
         "load_bearing": True,
         "description": (
             "At least one anchored arm holds a strict majority of seeds at retained_fraction "
             f">= {RETAINED_FRACTION_FLOOR}, beats the unconstrained arm's mean retained_fraction "
             f"by >= {RETENTION_ARM_MARGIN}, AND kept its plasticity (realised mean KL > "
             f"{KL_PLASTICITY_FLOOR} and trajectory peak >= "
             f"{1.0 + PEAK_IMPROVEMENT_MARGIN}x the install) -- so the retention is "
             "CONSOLIDATION and not FREEZING."
         ),
         "passed": c_load_bearing},
        {"name": "C_unconstrained_control_erodes_installed_competence",
         "load_bearing": False,
         "description": (
             "The unconstrained control does NOT hold the installed competence -- the contrast "
             "every anchored arm is read against. 780 measured 20.933 -> 11.667 here."
         ),
         "passed": bool(not off_retained)},
        {"name": "C_anchor_bound_dose_response",
         "load_bearing": False,
         "description": (
             "Realised mean KL to the frozen snapshot falls monotonically as kl_anchor_coef "
             f"rises, with a spread > {KL_DOSE_RESPONSE_MIN_SPREAD}. Measured WITHIN the anchored "
             "arms only: the control's guard value is a 0.0 sentinel, not a measured drift."
         ),
         "passed": anchor_bound},
    ]

    interpretation = {
        "label": label,
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "criteria": criteria,
        "criteria_non_degenerate": criteria_nd,
        "anchor_reachability": anchor_reachability,
        "interpretation_grid": [
            {"label": "substrate_not_ready_requeue", "outcome": "FAIL",
             "condition": (
                 "an anchor is sub-floor, OR post_bc_foraging_competence fails the 1.0 install "
                 "floor on a strict majority of seeds in ANY arm"
             ),
             "reading": (
                 "There is no installed prior to retain, so the run is UNINFORMATIVE about "
                 "retention. NEVER a retention verdict and never substrate_ceiling / "
                 "substrate_conditional / does_not_support. Requeue."
             )},
            {"label": "retention_consolidation_protects_competence", "outcome": "PASS",
             "condition": (
                 "some anchored arm retains on a strict majority of seeds, beats the "
                 "unconstrained arm by the declared margin, AND kept its plasticity (moved away "
                 "from the snapshot and improved above the install)"
             ),
             "reading": (
                 "An UPDATE CONSTRAINT is sufficient to preserve an acquired policy: the "
                 "consolidation/protection pathway MECH-457 posits is absent is constructible. "
                 "Read the winning coefficient as a lower bound on the constraint strength "
                 "required, not as a tuned optimum."
             )},
            {"label": "retention_by_policy_freezing", "outcome": "FAIL",
             "condition": (
                 "an anchored arm retains on a strict majority of seeds but its realised mean KL "
                 f"is at or below {KL_PLASTICITY_FLOOR} and its trajectory peak did not improve "
                 "above the install"
             ),
             "reading": (
                 "A retention/plasticity TRADEOFF, NOT a consolidation success and NOT the null. "
                 "Competence is preservable, but only by abolishing learning -- the anchor "
                 "pinned the policy to its snapshot rather than protecting it while it refined. "
                 "This is the trivially-retaining degenerate limit specific to an update "
                 "constraint (the critic swap in V3-EXQ-788 has no analogue), which is why this "
                 "branch exists here and not there. Re-pose at a weaker coefficient before "
                 "reading any retention verdict from this arm."
             )},
            {"label": "retention_consolidation_succeeded_then_decayed", "outcome": "FAIL",
             "condition": (
                 "an anchored arm's trajectory PEAK cleared the lift-competence target on a "
                 "strict majority of seeds, but its terminal retained_fraction fell below the "
                 "retention floor"
             ),
             "reading": (
                 "The manipulation SUCCEEDED and then DECAYED -- a retention-DYNAMICS finding, "
                 "not a null. Read the trajectory and the half-life, not the terminal scalar. "
                 "This is the branch V3-EXQ-780's grid lacked, which is why a manipulation that "
                 "succeeded above target was scored a null there and self-routed "
                 "bc_prior_not_the_axis (rejected on autopsy)."
             )},
            {"label": "retention_eroded_under_all_anchors", "outcome": "FAIL",
             "condition": (
                 "no anchored arm retains at any swept coefficient, and none shows the "
                 "succeeded-then-decayed or frozen signature"
             ),
             "reading": (
                 "THE DECLARED NULL. Anchoring does not preserve competence -> retention is NOT "
                 "a drift-protection problem. Removes H-retention-consolidation from the live "
                 "set; does NOT weaken MECH-457 (diagnostic). Read jointly with "
                 "H-retention-critic (V3-EXQ-788) and H-retention-auxiliary-decay "
                 "(V3-EXQ-789) -- the null is only informative about the UPDATE-CONSTRAINT "
                 "locus, and only across the swept coefficient range."
             )},
            {"label": "retention_grid_nondiscriminative", "outcome": "FAIL",
             "condition": (
                 "no arm passed its precondition gate, OR the anchor never bound (realised KL "
                 "shows no monotone dose-response across the swept coefficients), or the "
                 "readouts fall in none of the enumerated branches"
             ),
             "reading": (
                 "Not a refutation. Unscored; re-pose the measurement. An absent dose-response "
                 "specifically means the swept coefficients were all too weak to bind, which is "
                 "an instrument-range finding rather than a retention finding."
             )},
        ],
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown"},
        "per_arm_gate": gate["per_arm_gate"],
        "readiness": {
            "anchors_ready": anchors_ready,
            "install_took_all_arms": install_took_all,
            "anchor_bound_dose_response": anchor_bound,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
            "post_bc_worst_by_arm": {
                a: per_arm_retention[a]["post_bc_foraging_competence_worst"] for a in BOOT_ARMS
            },
            "post_bc_worst_seed_by_arm": {
                a: per_arm_retention[a]["post_bc_worst_seed"] for a in BOOT_ARMS
            },
            # P0 READINESS-ASSERT, restated flat and numerically so it is readable without
            # reconstructing the per-arm gate. SAME statistic the load-bearing criterion routes
            # on (foraging_competence), measured on the WORST cell across all arms.
            "readiness_assert": {
                "name": "post_bc_install_took_worst_cell",
                "kind": "readiness",
                "description": (
                    "The BC install must have TAKEN before RL on every arm: post-BC / pre-RL "
                    "foraging_competence of the worst cell clears the 1.0 install floor. Below "
                    "it there is no installed prior to retain and the run self-routes "
                    "substrate_not_ready_requeue."
                ),
                "control": "post_bc_foraging_competence, worst cell over all arms x all seeds",
                "direction": "lower",
                "measured": round(min(
                    float(per_arm_retention[a]["post_bc_foraging_competence_worst"])
                    for a in BOOT_ARMS
                ), 6) if all(per_arm_retention[a]["n_cells"] for a in BOOT_ARMS) else 0.0,
                "threshold": float(POST_BC_INSTALL_FLOOR),
                "met": install_took_all,
            },
        },
        "anchor_dose_response": {
            "description": (
                "Realised mean KL to the frozen installed-policy snapshot, by anchor "
                "coefficient. Measured WITHIN the anchored arms ONLY: the unconstrained "
                "control constructs no PolicyKLAnchor, so its guard value is a hard-coded 0.0 "
                "sentinel rather than a measured drift, and comparing against it reads "
                "backwards. A monotone decrease with rising coefficient is the only available "
                "proof that the anchor actually BOUND."
            ),
            "kl_by_coef": [
                {"kl_anchor_coef": c, "mean_policy_kl_to_anchor_mean": v}
                for c, v in anchored_kl_by_coef
            ],
            "monotone_decreasing": kl_monotone_decreasing,
            "spread": kl_spread,
            "min_spread_threshold": KL_DOSE_RESPONSE_MIN_SPREAD,
            "anchor_bound": anchor_bound,
            "unconstrained_guard_value_is_sentinel": True,
        },
        "headline": {
            "consolidation_protects_installed_competence": c_load_bearing,
            "consolidating_arms": consolidating_arms,
            "frozen_arms": frozen_arms,
            "succeeded_then_decayed_arms": decayed_arms,
            "eroded_under_all_anchors": eroded_under_all,
            "retained_fraction_mean_unconstrained": off_frac,
            "retained_fraction_mean_by_anchored_arm": {
                a: per_arm_retention[a]["retained_fraction_mean"] for a in ANCHORED_ARMS
            },
            "retained_fraction_margin_by_anchored_arm": {
                a: _margin_over_off(a) for a in ANCHORED_ARMS
            },
            "mean_policy_kl_by_anchored_arm": {
                a: per_arm_retention[a]["mean_policy_kl_to_anchor_mean"] for a in ANCHORED_ARMS
            },
            "anchor_bound_dose_response": anchor_bound,
            "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
            "retention_arm_margin": RETENTION_ARM_MARGIN,
            "kl_plasticity_floor": KL_PLASTICITY_FLOOR,
            "peak_improvement_margin": PEAK_IMPROVEMENT_MARGIN,
            "peak_success_target": PEAK_SUCCESS_TARGET,
            "retention_probe_every": int(probe_every),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
        },
        "per_arm_verdict": per_arm_verdict,
        "per_arm_retention": per_arm_retention,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "post_bc_install_floor": float(POST_BC_INSTALL_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
            "post_bc_780_reference_raw_view": 20.933,
            "unconstrained_terminal_780_reference_raw_view": 11.667,
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
        "anchor_dose_response": result["anchor_dose_response"],
        "headline": result["headline"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_verdict": result["per_arm_verdict"],
        "per_arm_retention": result["per_arm_retention"],
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
                "The OFF arm (" + OFF_ARM + ") is emitted reuse-ELIGIBLE with the SHARED lineage "
                "slice (experiments/_lib/baselines/mech457_retention.off_path_config_slice) and "
                "include_driver_script_in_hash=False, so a sibling retention leg can reuse the "
                "minted cell across its different driver. NO reuse was attempted FROM "
                "V3-EXQ-788's mint, deliberately: 788 ran at substrate_hash 953cb8d0... and the "
                "enabling commit ree-v3 399b17caed edited three files under experiments/_lib/** "
                "(mech457_explorer_classes.py, mech457_bootstrap_explorer.py, "
                "baselines/mech457_retention.py), all inside the substrate glob, so the "
                "fingerprint CORRECTLY refuses that cell. A false MISS is the safe direction of "
                "the governing asymmetry. This mint is at the post-399b17caed substrate_hash and "
                "is reusable by V3-EXQ-789 if that leg runs on the same substrate."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The post-installation competence TRAJECTORY (not terminal competence) of a "
            "BC-installed raw_view policy across the reference RL refinement, probed every "
            f"{cfg['retention_probe_every']} episodes, under an UNCONSTRAINED update vs a policy "
            f"KL anchor to the INSTALLED policy at coefficients {list(KL_ANCHOR_COEFS)}. "
            "Statistic: retained_fraction = terminal trajectory competence / installed (post-BC) "
            "competence, with the trajectory PEAK and the realised mean KL to the frozen "
            "snapshot read alongside it. PASS: some anchored arm holds retained_fraction >= "
            f"{RETAINED_FRACTION_FLOOR} on a strict majority of seeds, beats the unconstrained "
            f"arm's mean by >= {RETENTION_ARM_MARGIN}, AND kept its plasticity (realised mean KL "
            f"> {KL_PLASTICITY_FLOOR} and peak >= {1.0 + PEAK_IMPROVEMENT_MARGIN}x the install) "
            "-- so a frozen policy cannot pass by retaining trivially. Readiness: "
            "post_bc_foraging_competence clears the 1.0 install floor on the worst seed of every "
            "arm (780 measured 20.933 on raw_view, 3/3 taking); an install that did not take "
            "self-routes substrate_not_ready_requeue."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 RETENTION leg H-retention-consolidation, pre-registered under "
            "question 'competence_floor' in hypothesis_space_registry.v1.json. DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy (read "
            "jointly with H-retention-critic V3-EXQ-788 and H-retention-auxiliary-decay "
            "V3-EXQ-789). MANIPULATION = the UPDATE CONSTRAINT ONLY (use_policy_kl_anchor / "
            "kl_anchor_coef); the value estimator is untouched (use_distributional_critic False "
            "on every arm; the KL term puts exactly zero gradient on value_head/value_bins) and "
            "the auxiliary is the constant baseline 0.5 on every arm. The anchor is a frozen "
            "deep copy of the LEARNER'S OWN post-install weights, never the demonstrator, which "
            "is what keeps it disjoint from V3-EXQ-789. MOTIVATION: V3-EXQ-780 raw_view post-BC "
            "20.933 eroded to 11.667 under unconstrained refinement with 3/3 seeds having taken "
            "the install. raw_view ONLY (780: 20.933 with 3/3 taking on raw_view vs 0.583 with "
            "0/3 on z_world -- a retention question is unanswerable where the install does not "
            "take). Reference build 128-wide / 3x budget / z_world detached / credit-replay 3 / "
            "topk 32 via baselines.reference_config -- NOT the 769-falsified 256/5x regression. "
            "COEFFICIENT SWEEP 0.03/0.10/0.30 brackets the entropy-beta band (0.10 -> 0.03), the "
            "only other distribution-level regulariser in the same loss, and is three-valued "
            "because the cross-coef KL dose-response is the ONLY valid proof the anchor bound: "
            "the control's mean_policy_kl_to_anchor_recent is a hard-coded 0.0 sentinel, not a "
            "measured drift. DECLARED NULL: anchoring does not preserve competence -> retention "
            "is NOT a drift-protection problem (label retention_eroded_under_all_anchors); this "
            "does NOT weaken MECH-457, which stays candidate/v3_pending."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-792 MECH-457 GOV-FANOUT-1 H-retention-consolidation"
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
    else:
        seeds = list(fan.SEEDS)
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)  # 3000 -- reference budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        probe_every = RETENTION_PROBE_EVERY

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
                            probe_every=probe_every)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representation": REF_REPRESENTATION,
        "on_budget_episodes": on_budget,
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "retention_probe_every": probe_every,
        "bc_warmstart_episodes": int(baselines.BC_WARMSTART_EPISODES),
        "bc_aux_coef": float(baselines.BC_AUX_COEF_BASELINE),
        "bc_demonstrator": "local_view_greedy",
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": baselines.REF_CREDIT_PASSES,
        "ref_credit_topk": baselines.REF_CREDIT_TOPK,
        "ref_cotrain_encoder": baselines.REF_COTRAIN_ENCODER,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA, "bc_lr": fan.BC_LR,
        "kl_anchor_coefs": list(KL_ANCHOR_COEFS),
        "kl_anchor_coef_scale_reference": {
            "entropy_beta_start": float(boot.ON_ENTROPY_BETA_START),
            "entropy_beta_end": float(boot.ON_ENTROPY_BETA_END),
            "note": (
                "The KL penalty enters the SAME policy loss as the entropy bonus, the only "
                "other distribution-level regulariser on the same logits, so the swept "
                "coefficients bracket the entropy-beta band and extend one step beyond it."
            ),
        },
        **{
            f"{k}_config": _make_cfg(k, on_budget, probe_every).as_slice()
            for k in CFG_KINDS
        },
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "retention_arm_margin": RETENTION_ARM_MARGIN,
        "kl_plasticity_floor": KL_PLASTICITY_FLOOR,
        "peak_improvement_margin": PEAK_IMPROVEMENT_MARGIN,
        "kl_dose_response_min_spread": KL_DOSE_RESPONSE_MIN_SPREAD,
        "peak_success_target": PEAK_SUCCESS_TARGET,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 retention (H-retention-consolidation)",
        "hypothesis_id": "H-retention-consolidation",
        "hypothesis_question": "competence_floor",
        "sibling_legs": "V3-EXQ-788 (H-retention-critic), V3-EXQ-789 (H-retention-auxiliary-decay)",
        "substrate_commit": "399b17caed (mech457_policy_kl_anchor)",
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
        f"install_took_all={result['readiness']['install_took_all_arms']} "
        f"anchor_bound={result['readiness']['anchor_bound_dose_response']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for arm_id in BOOT_ARMS:
        r = result["per_arm_retention"][arm_id]
        print(
            f"  {arm_id}: coef={r['kl_anchor_coef']} "
            f"post_bc_worst={r['post_bc_foraging_competence_worst']} "
            f"(seed={r['post_bc_worst_seed']}) install_took={r['n_seeds_install_took']}/"
            f"{r['n_cells']} readings_worst={r['n_trajectory_readings_worst']} "
            f"retained_frac_mean={r['retained_fraction_mean']} "
            f"kl_mean={r['mean_policy_kl_to_anchor_mean']} "
            f"plastic={r['n_seeds_retained_plasticity']}/{r['n_cells']} "
            f"peak_mean={r['trajectory_peak_mean']} "
            f"terminal={r['terminal_forage_mean']}", flush=True,
        )
    print(
        f"  dose_response: kl_by_coef={result['anchor_dose_response']['kl_by_coef']} "
        f"monotone={result['anchor_dose_response']['monotone_decreasing']} "
        f"spread={result['anchor_dose_response']['spread']}", flush=True,
    )
    print(
        f"  consolidating={hl['consolidating_arms']} frozen={hl['frozen_arms']} "
        f"decayed={hl['succeeded_then_decayed_arms']} "
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
