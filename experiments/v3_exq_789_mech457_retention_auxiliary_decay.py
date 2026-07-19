#!/opt/local/bin/python3
"""V3-EXQ-789 -- MECH-457 GOV-FANOUT-1 RETENTION leg, H-retention-auxiliary-decay: is the
imitation AUXILIARY out-competed by the RL objective over training?

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PRE-REGISTERED as hypothesis H-retention-auxiliary-decay under question qid "competence_floor"
in REE_assembly/evidence/planning/hypothesis_space_registry.v1.json.

THE HYPOTHESIS. The installed prior is not HELD rather than not PROTECTED: the imitation
auxiliary that installs competence is progressively out-competed by the RL objective, so the
installed policy decays back toward the RL optimum. The pair partner V3-EXQ-788
(H-retention-critic) asks the complementary question -- whether a better VALUE ESTIMATOR lets
the installed policy be REFINED rather than eroded. Read the two legs jointly.

MOTIVATING EVIDENCE. V3-EXQ-780 ran a CONSTANT bc_aux_coef=0.5 throughout and still lost the
install: 20.933 post-BC -> 11.667 terminal. So a persistent-but-fixed auxiliary is already known
INSUFFICIENT. What is untested is the SCHEDULE dimension: whether auxiliary persistence buys any
retention TIME at all, measured as a decay rate rather than as a terminal level.

DESIGN. BC-install the raw_view policy to the ~20.9 competence point, then sweep the auxiliary's
PERSISTENCE and measure the competence HALF-LIFE -- a trajectory statistic by construction.
Exactly three boot arms, all BC-installed, all at the reference build, differing ONLY in the
bc_aux_* triple:
  * retaux_constant  -- bc_aux_coef=0.5 held throughout (reproduces the 780 condition, and is
                        also V3-EXQ-788's control: this is the SHARED lineage baseline).
  * retaux_annealed  -- bc_aux_coef 0.5 -> ~0.0 over the first bc_aux_anneal_fraction of the
                        budget, via bc_aux_coef_end + bc_aux_anneal_fraction.
  * retaux_off       -- bc_aux_coef=0.0: no persistent auxiliary at all after the install.

THE ANNEAL USES linear_anneal, NOT warm_then_anneal. This is pre-registered and load-bearing.
warm_then_anneal is parameterised by the SHARED warm_start_fraction, which would couple BC
persistence to the EXPLORATION anneal and confound this leg's single intervention. The substrate
wires bc_aux_coef / bc_aux_coef_end / bc_aux_anneal_fraction to linear_anneal for us (see
mech457_bootstrap_explorer.train_bootstrap_explorer), and the BC schedule is deliberately OUTSIDE
the mode_gate exclusion, so no schedule is hand-rolled here.

ANTI-ALIAS (load-bearing). use_distributional_critic stays False on ALL THREE arms. That knob is
V3-EXQ-788's locus (H-retention-critic); a leg that moved both would read as neither. Everything
outside the bc_aux_* triple -- capacity (128-wide), budget (3x), credit replay 3 / topk 32, the
developmental drive anneal, the install, the probe cadence -- is supplied by
baselines.reference_config() and is IDENTICAL across the three arms by construction. The
769-FALSIFIED capacity regression (boot.make_on_config / ON_ACTOR_CRITIC_HIDDEN=256 /
ON_BUDGET_MULTIPLIER=5) is deliberately NOT used anywhere in this driver.

EVERY ARM IS BUILT THROUGH experiments/_lib/baselines/mech457_retention.py -- build_off_arm()
forwards use_distributional_critic from cfg, so it builds ANY arm of this lineage correctly and
is used unconditionally. That is what makes this leg's shared baseline cell agree with
V3-EXQ-788's by CONSTRUCTION rather than by two drivers happening to spell the same config.

THE DV IS A HALF-LIFE, NOT A TERMINAL SCALAR. cfg.retention_probe_every wires the substrate's
non-perturbing mid-training competence probe (train_a2c snapshots and restores the torch/numpy/
random streams around every reading, so measurement neutrality is a substrate guarantee). At the
same 250-episode cadence V3-EXQ-788 uses -- so the two legs' trajectories are directly comparable
-- that is 12 readings per (arm x seed) over the 3000-episode budget, each recorded in full. The
PRIMARY DV is baselines.competence_half_life(): episodes until competence first falls below half
the installed value. It returns None in TWO substantively different cases -- the install was ~0
(half-life UNDEFINED) and competence NEVER halved (DID NOT DECAY) -- so this driver BRANCHES on
None into an explicit per-cell `half_life_status` and never coerces it to a shared sentinel.

SCHEDULE VERIFICATION (the trap this leg must not fall into). The substrate warns that the
schedule drives the auxiliary's GUARD as well as its weight: an annealed arm passes
bc_aux_coef=0.0 with a nonzero schedule, so a guard reading the CONSTANT would silently produce
an OFF arm labelled ANNEALED. Symmetrically, an annealed arm whose schedule silently stayed flat
is indistinguishable from the CONSTANT arm. Both are caught here as numeric PRECONDITIONS scoped
(applies_to) to the annealed arm, read off the realised bc_aux_coef_first / bc_aux_coef_last that
train_a2c returns and that this manifest emits per cell -- so a non-moving or collapsed schedule
is CAUGHT rather than adjudicated.

INTERPRETATION GRID (five enumerated branches; the "manipulation succeeded and then decayed"
branch is mandatory and is the one V3-EXQ-780 lacked):
  * substrate_not_ready_requeue                 -- the install did NOT take (or an anchor is
      sub-floor). UNINFORMATIVE about retention. NEVER a retention verdict, and never
      substrate_ceiling / substrate_conditional / does_not_support / *_nondiscriminative /
      *_unmeetable.
  * half_life_extended_by_persistence           -- PASS / THE HYPOTHESIS. Auxiliary persistence
      buys retention TIME: half-life ordered constant >= annealed >= off, with constant beating
      off by the declared margin.
  * retention_auxiliary_succeeded_then_decayed  -- the most-persistent arm ROSE to at/above the
      lift-competence target at its trajectory PEAK and then FELL below the retention floor. The
      manipulation worked and then decayed: a retention-DYNAMICS finding, not a null.
  * half_life_invariant_to_schedule             -- THE DECLARED NULL. Half-life is invariant to
      the auxiliary schedule -> the prior is NOT being out-competed by the RL objective.
  * retention_auxiliary_grid_nondiscriminative  -- no arm passed its gate, the arms do not
      separate, or a half-life is undefined where it must be defined. Not a refutation.

ROUTING CONSUMES THE DECLARED COVARIATES, NOT ONLY THE TERMINAL CRITERION.
post_bc_foraging_competence is LOAD-BEARING in the routing (it selects the requeue branch before
any retention branch can be reached), the trajectory SHAPE (peak vs terminal) selects between the
extended / succeeded-then-decayed branches, and the REALISED schedule (bc_aux_coef_first/_last)
gates the annealed arm. This is the exact defect that sank V3-EXQ-780: post_bc_foraging_competence
was present and declared there, but its grid enumerated only a ~0 null, so a manipulation that
succeeded ABOVE target was scored a null and self-routed bc_prior_not_the_axis (rejected on
autopsy).

MULTI-ARM GATE. experiments/_lib/precondition_gate.py -- per-arm gates aggregated with
aggregate_arm_gates so non_degenerate = ANY arm green. One arm's unmet precondition must NEVER
vacate another's (the V3-EXQ-785 defect).

MINT (mint-as-you-go). The shared baseline arm retaux_constant emits its cell reuse-ELIGIBLE with
the shared lineage slice (baselines.off_path_config_slice) and include_driver_script_in_hash=False,
so it interchanges with the cell V3-EXQ-788 mints for its scalar control across the two DIFFERENT
drivers. The flag must match on both sides or reuse is refused; 788 sets it the same way. First
leg to run mints; second reuses. No separate baseline-only mint job.

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

EXPERIMENT_TYPE = "v3_exq_789_mech457_retention_auxiliary_decay"
QUEUE_ID = "V3-EXQ-789"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# --- Probe cadence -- MATCHED TO V3-EXQ-788 so the two legs' trajectories are comparable -----
# 3000-episode reference budget / 250 = 12 readings per (arm x seed). Declared as a module
# constant (not derived at call time) because it is fingerprint-declared in the config_slice:
# a probed and an unprobed cell are bit-identical COMPUTATIONS but are not interchangeable
# ARTIFACTS, since only one carries the trajectory a consumer reads.
RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# --- Pre-registered auxiliary-persistence schedule (declared; never derived from a run) ------
BC_AUX_COEF_START = baselines.BC_AUX_COEF_BASELINE   # 0.5 -- the persistent auxiliary 780 ran
BC_AUX_COEF_ANNEAL_END = 0.0                         # annealed arm decays the auxiliary to zero
BC_AUX_ANNEAL_FRACTION = 0.5                         # over the first half of the RL budget
BC_AUX_COEF_OFF = 0.0                                # no persistent auxiliary at all

# --- Pre-registered retention thresholds (declared; never derived from the run) --------------
RETAINED_FRACTION_FLOOR = 0.5          # >= half the installed competence survives refinement
# The half-life margin is expressed as a FRACTION OF THE BUDGET so it scales with --episodes and
# stays meaningful under --dry-run; the realised episode margin is recorded in the manifest.
HALF_LIFE_MARGIN_FRACTION_OF_BUDGET = 0.15
PEAK_SUCCESS_TARGET = baselines.LIFT_COMPETENCE_TARGET   # ~13.05 res/ep
MIN_TRAJECTORY_READINGS = 1.5          # need >= 2 readings for a SHAPE (floor, strict >)
# Schedule-verification floors, both scoped to the annealed arm (see the module docstring).
BC_AUX_SCHEDULE_MOVE_FLOOR = 0.1       # realised (first - last) must be a real move
BC_AUX_SCHEDULE_START_FLOOR = 0.25     # realised first must be an ACTIVE auxiliary, not ~0

# The install / reference build. Re-exported from the shared lineage module so this driver
# cannot drift from it.
REF_REPRESENTATION = baselines.REF_REPRESENTATION          # raw_view ONLY
REF_ACTOR_CRITIC_HIDDEN = baselines.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = baselines.REF_BUDGET_MULTIPLIER      # 3x
POST_BC_INSTALL_FLOOR = baselines.POST_BC_INSTALL_FLOOR      # 1.0 competence floor

CFG_KINDS: Tuple[str, ...] = ("constant", "annealed", "off")
BASELINE_KIND = "constant"      # the SHARED lineage baseline (== 780 condition == 788 control)


def _arm_id(cfg_kind: str) -> str:
    return f"retaux_{cfg_kind}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(k) for k in CFG_KINDS)
BASELINE_ARM = _arm_id(BASELINE_KIND)
ANNEALED_ARM = _arm_id("annealed")
OFF_AUX_ARM = _arm_id("off")
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


def _aux_params(cfg_kind: str) -> Tuple[float, Optional[float], float]:
    """(bc_aux_coef, bc_aux_coef_end, bc_aux_anneal_fraction) for one arm.

    The ONLY thing that varies across the three boot arms. end=None is the substrate's
    constant-coefficient path (no schedule at all), which is what the CONSTANT and OFF arms want.
    """
    if cfg_kind == "constant":
        return BC_AUX_COEF_START, None, 0.0
    if cfg_kind == "annealed":
        return BC_AUX_COEF_START, BC_AUX_COEF_ANNEAL_END, BC_AUX_ANNEAL_FRACTION
    if cfg_kind == "off":
        return BC_AUX_COEF_OFF, None, 0.0
    raise ValueError(f"unknown cfg_kind {cfg_kind!r}")


def _declared_schedule_drop(n_episodes: int) -> float:
    """Design-time realised drop of the annealed arm's schedule over its declared budget.

    Computed with the SAME linear_anneal the substrate wires, at the first and last episode
    indices the training loop will actually use. This is the structural proof that backs the
    schedule-moved precondition: a budget too short for the anneal to bite is caught before any
    compute is spent rather than being adjudicated afterwards.
    """
    n = max(1, int(n_episodes))
    first = boot.linear_anneal(BC_AUX_COEF_START, BC_AUX_COEF_ANNEAL_END,
                               BC_AUX_ANNEAL_FRACTION, 0, n)
    last = boot.linear_anneal(BC_AUX_COEF_START, BC_AUX_COEF_ANNEAL_END,
                              BC_AUX_ANNEAL_FRACTION, n - 1, n)
    return float(first - last)


# ---------------------------------------------------------------------------------------
# Precondition specs (multi-arm gate). Every entry carries numeric measured + threshold so
# build_experiment_indexes.py can recompute `met`. All six are FLOORS (direction defaults to
# "lower"); none is a ceiling-style bound, so none needs direction="upper".
# ---------------------------------------------------------------------------------------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="local_view_greedy_clears_floor_at_d3",
        description=(
            "LocalViewGreedyPolicy reading the SAME 5x5 resource_field_view forages above the "
            "1.0 competence floor at D3 -- the positive control that the env is solvable from "
            "the local view. Below-floor means the substrate/env is not ready, NOT that the "
            "auxiliary schedule failed."
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
            "there is no installed prior to decay, the half-life is UNDEFINED rather than "
            "short, and the run self-routes substrate_not_ready_requeue rather than any "
            "retention verdict."
        ),
        control="post-BC / pre-RL foraging_competence of this arm's worst seed vs the 1.0 floor",
        threshold=float(POST_BC_INSTALL_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="competence_trajectory_readings",
        description=(
            "The retention DV is a HALF-LIFE read off a TRAJECTORY, so the arm's worst cell must "
            "carry at least two probe readings -- one reading has no shape, cannot locate a "
            "half-life, and cannot separate 'retained' from 'succeeded then decayed'."
        ),
        control="number of mid-training competence probe readings in this arm's worst cell",
        threshold=float(MIN_TRAJECTORY_READINGS),
        kind="measurability",
        # Design-time proof: cadence and budget are both pre-registered, so an unsatisfiable
        # probe schedule is caught before any compute is spent.
        structural_max=lambda ctx: float(
            int(ctx["n_episodes"]) // max(1, int(ctx["probe_every"]))
        ),
    ),
    PreconditionSpec(
        name="bc_aux_schedule_moved",
        description=(
            "THE SCHEDULE-VERIFICATION PRECONDITION. The annealed arm's REALISED auxiliary "
            "weight must actually have moved: bc_aux_coef_first - bc_aux_coef_last, read off "
            "train_a2c's returned values on this arm's worst cell. The substrate warns that the "
            "schedule drives the auxiliary's GUARD as well as its weight, so an annealed arm "
            "whose schedule silently stayed flat is INDISTINGUISHABLE from the constant arm and "
            "would be adjudicated as a null it never tested. Caught here numerically instead."
        ),
        control=(
            "realised bc_aux_coef_first minus bc_aux_coef_last on the annealed arm's worst cell"
        ),
        threshold=float(BC_AUX_SCHEDULE_MOVE_FLOOR),
        kind="manipulation_active",
        applies_to=lambda ctx: bool(ctx["is_annealed"]),
        applies_note=(
            "annealed arm only -- the constant and off arms are DEFINED by holding a fixed "
            "coefficient (0.5 and 0.0), so demanding that their schedule move would make their "
            "gates structurally un-passable and collapse the three-arm design"
        ),
        structural_max=lambda ctx: (
            _declared_schedule_drop(int(ctx["n_episodes"])) if ctx["is_annealed"] else None
        ),
    ),
    PreconditionSpec(
        name="bc_aux_schedule_starts_active",
        description=(
            "The annealed arm's REALISED auxiliary must START active: bc_aux_coef_first on this "
            "arm's worst cell. An annealed arm that collapsed to zero from the outset is "
            "INDISTINGUISHABLE from the off arm, which is the mirror image of the flat-schedule "
            "failure above -- the two together pin the annealed arm strictly between its "
            "neighbours rather than letting it alias onto either."
        ),
        control="realised bc_aux_coef_first on the annealed arm's worst cell",
        threshold=float(BC_AUX_SCHEDULE_START_FLOOR),
        kind="manipulation_active",
        applies_to=lambda ctx: bool(ctx["is_annealed"]),
        applies_note=(
            "annealed arm only -- the off arm is DEFINED by a zero auxiliary, so asserting an "
            "active starting weight there would make its gate structurally un-passable"
        ),
        structural_max=lambda ctx: (
            float(BC_AUX_COEF_START) if ctx["is_annealed"] else None
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
    """The shared reference RL-refinement config, varying ONLY the bc_aux_* triple.

    Everything else -- actor_critic_hidden, budget, credit replay, the drive anneal, the install,
    the probe cadence -- comes from baselines.reference_config() and is therefore IDENTICAL
    across the three arms by construction. use_distributional_critic is left at its False
    default on EVERY arm: that knob is V3-EXQ-788's locus (H-retention-critic), and a leg moving
    both would read as neither.
    """
    coef, coef_end, frac = _aux_params(cfg_kind)
    return baselines.reference_config(
        on_budget,
        bc_aux_coef=coef,
        bc_aux_coef_end=coef_end,
        bc_aux_anneal_fraction=frac,
        retention_probe_every=int(probe_every),
    )


def _config_slice(cfg_kind: str, env_kwargs: Dict[str, Any], eval_eps: int, steps: int,
                  on_budget: int, probe_every: int) -> Dict[str, Any]:
    """All three arms declare the SHARED lineage slice; the two non-baseline arms then override
    their own arm_id and fold in their (bc_aux_*) config. Anchoring all of them on
    off_path_config_slice makes the baseline-cell match with V3-EXQ-788 structural rather than
    accidental."""
    base = baselines.off_path_config_slice(
        env_kwargs, eval_eps=int(eval_eps), steps=int(steps),
        on_budget=int(on_budget), retention_probe_every=int(probe_every),
    )
    if cfg_kind == BASELINE_KIND:
        return base
    base["arm_id"] = _arm_id(cfg_kind)
    base["kind"] = "mech457_retention_auxiliary_decay_treatment"
    base.update(_make_cfg(cfg_kind, on_budget, probe_every).as_slice())
    return base


def _half_life_status(trajectory: List[Dict[str, Any]], post_bc: float,
                      half_life: Optional[float]) -> str:
    """Branch on competence_half_life()'s None -- it means TWO different things.

    "undefined_no_install": the install was ~0 (or nothing was probed), so the ratio the
    half-life is defined against does not exist. NOT a fast decay.
    "did_not_decay": competence never fell below half the installed value over the whole budget.
    NOT a large half-life measurement either -- it is right-CENSORED at the budget.
    Coercing either to a shared numeric sentinel is exactly what the substrate warns against.
    """
    if not trajectory or float(post_bc) <= 0.0:
        return "undefined_no_install"
    if half_life is None:
        return "did_not_decay"
    return "decayed"


def _run_boot_cell(cfg_kind: str, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind)
    cfg = _make_cfg(cfg_kind, on_budget, probe_every)
    # build_off_arm forwards use_distributional_critic from cfg, so it builds ANY arm of this
    # lineage correctly -- used unconditionally for all three arms (no direct make_rep path).
    rep_agent = baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)

    # Phase 1 -- BC install to the ~20.9 competence point, then measure whether it TOOK.
    # IDENTICAL on all three arms: the sweep is over what happens AFTER the install.
    install = baselines.install_bc_prior(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    post_bc = float(install["post_bc_foraging_competence"])

    # Phase 2 -- the SAME RL refinement under this arm's auxiliary PERSISTENCE, probed on a
    # cadence. The off arm passes demo=None: with bc_aux_coef 0.0 the auxiliary term is already
    # inert, and withholding the demonstrator also withholds its per-step act() call, so "no
    # persistent auxiliary at all" is literal rather than merely zero-weighted.
    probe_fn = baselines.make_probe_fn(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    guard = baselines.train_off_arm(
        rep_agent, seed, env_kwargs, steps=steps, arm_label=arm_id, cfg=cfg,
        demo=(None if cfg_kind == "off" else install["demo"]), probe_fn=probe_fn,
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
    hl_status = _half_life_status(trajectory, post_bc, half_life)

    coef_first = guard.get("bc_aux_coef_first")
    coef_last = guard.get("bc_aux_coef_last")
    schedule_drop = (
        round(float(coef_first) - float(coef_last), 6)
        if (coef_first is not None and coef_last is not None) else None
    )

    row["post_bc_foraging_competence"] = round(post_bc, 6)
    row["install_took"] = bool(install["install_took"])
    row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
    row["bc_aux_kind"] = cfg_kind
    row["bc_aux_coef_declared"] = float(cfg.bc_aux_coef)
    row["bc_aux_coef_end_declared"] = (
        None if cfg.bc_aux_coef_end is None else float(cfg.bc_aux_coef_end)
    )
    row["bc_aux_anneal_fraction_declared"] = float(cfg.bc_aux_anneal_fraction)
    row["use_distributional_critic"] = bool(cfg.use_distributional_critic)   # False on ALL arms
    # FULL per-seed trajectory -- never collapsed to a terminal scalar.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    row["retained_fraction"] = retained
    row["competence_half_life_episodes"] = half_life         # None is MEANINGFUL -- see status
    row["competence_half_life_status"] = hl_status
    row["peak_cleared_success_target"] = bool(peak >= PEAK_SUCCESS_TARGET)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["mean_bc_aux_action_match_recent"] = float(guard.get("mean_bc_aux_action_match_recent", 0.0))
    # REALISED schedule -- the schedule-verification evidence, emitted per cell.
    row["bc_aux_coef_first"] = coef_first
    row["bc_aux_coef_last"] = coef_last
    row["bc_aux_schedule_drop"] = schedule_drop
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


def _arm_contexts(on_budget: int, probe_every: int) -> List[Dict[str, Any]]:
    ctxs: List[Dict[str, Any]] = []
    for k in CFG_KINDS:
        coef, coef_end, frac = _aux_params(k)
        ctxs.append({
            "id": _arm_id(k),
            "cfg_kind": k,
            "is_annealed": (k == "annealed"),
            "bc_aux_coef": float(coef),
            "bc_aux_coef_end": coef_end,
            "bc_aux_anneal_fraction": float(frac),
            "n_episodes": int(on_budget),
            "probe_every": int(probe_every),
        })
    return ctxs


def run_experiment(seeds: List[int], on_budget: int, eval_eps: int, steps: int,
                   probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-retention-auxiliary-decay "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep={REF_REPRESENTATION}, ON_budget={on_budget}, probe_every={probe_every}, "
        f"eval={eval_eps}, steps={steps}; ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, "
        f"budget_mult={REF_BUDGET_MULTIPLIER}; manipulation=bc_aux persistence ONLY "
        f"[constant {BC_AUX_COEF_START} / annealed {BC_AUX_COEF_START}->"
        f"{BC_AUX_COEF_ANNEAL_END} over {BC_AUX_ANNEAL_FRACTION} via linear_anneal / off "
        f"{BC_AUX_COEF_OFF}])",
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
    declared_drop = _declared_schedule_drop(on_budget)
    print(
        f"declared annealed schedule drop over {on_budget} episodes: {round(declared_drop, 6)} "
        f"(floor {BC_AUX_SCHEDULE_MOVE_FLOOR}; linear_anneal, NOT warm_then_anneal)", flush=True,
    )

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    half_life_margin = float(HALF_LIFE_MARGIN_FRACTION_OF_BUDGET) * float(on_budget)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        is_boot = arm_id in BOOT_ARMS
        is_baseline = (arm_id == BASELINE_ARM)
        if is_boot:
            slice_cfg = _config_slice(cfg_kind, env_kwargs, eval_eps, steps, on_budget,
                                      probe_every)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID,
                         "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        # The SHARED baseline arm (and the anchors) mint cross-driver-reusable -- no driver
        # script in the hash -- so V3-EXQ-788's scalar control cell and this leg's constant arm
        # interchange. The two treatment arms are leg-specific and keep the driver in their hash.
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=not (is_baseline or not is_boot)) as cell:
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

    # ---- per-arm retention readouts (half-life shaped, worst-cell reported) ----------------
    def _cells_for(arm_id: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == arm_id]

    def _worst(cells: List[Dict[str, Any]], key: str, default: float) -> Tuple[float, Any]:
        if not cells:
            return float(default), None
        vals = [c for c in cells if c.get(key) is not None]
        if not vals:
            return float(default), None
        best = min(vals, key=lambda c: float(c.get(key, default)))
        return float(best.get(key, default)), best.get("seed")

    per_arm_retention: Dict[str, Any] = {}
    for cfg_kind in CFG_KINDS:
        arm_id = _arm_id(cfg_kind)
        cells = _cells_for(arm_id)
        post_bc_worst, post_bc_worst_seed = _worst(cells, "post_bc_foraging_competence", 0.0)
        n_read_worst, n_read_worst_seed = _worst(cells, "n_trajectory_readings", 0.0)
        drop_worst, drop_worst_seed = _worst(cells, "bc_aux_schedule_drop", 0.0)
        first_worst, first_worst_seed = _worst(cells, "bc_aux_coef_first", 0.0)
        statuses = [str(c.get("competence_half_life_status", "undefined_no_install"))
                    for c in cells]
        n_decayed = int(sum(1 for s in statuses if s == "decayed"))
        n_did_not_decay = int(sum(1 for s in statuses if s == "did_not_decay"))
        n_undefined = int(sum(1 for s in statuses if s == "undefined_no_install"))
        # EFFECTIVE half-life for the ORDERING comparison only, with the "did not decay" cells
        # RIGHT-CENSORED at the budget (the standard survival treatment of a non-event) and the
        # "undefined" cells left as None. This is a declared branch on the two None meanings,
        # not a coercion of them into one sentinel: both counts are reported alongside, and a
        # single undefined cell is enough to send the arm to the non-discriminative branch.
        effective: List[Optional[float]] = []
        for c in cells:
            st = str(c.get("competence_half_life_status", "undefined_no_install"))
            if st == "decayed":
                effective.append(float(c.get("competence_half_life_episodes")))
            elif st == "did_not_decay":
                effective.append(float(on_budget))
            else:
                effective.append(None)
        eff_num = [v for v in effective if v is not None]
        eff_mean = round(float(sum(eff_num) / len(eff_num)), 6) if eff_num else None
        retained_vals = [c.get("retained_fraction") for c in cells]
        retained_num = [float(v) for v in retained_vals if v is not None]
        retained_mean = (
            round(float(sum(retained_num) / len(retained_num)), 6) if retained_num else None
        )
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        n_peak_cleared = int(sum(1 for c in cells
                                 if bool(c.get("peak_cleared_success_target", False))))
        n_retained = int(sum(1 for v in retained_num if v >= RETAINED_FRACTION_FLOOR))
        n_cells = len(cells)
        per_arm_retention[arm_id] = {
            "arm_id": arm_id,
            "bc_aux_kind": cfg_kind,
            "bc_aux_coef_declared": float(_aux_params(cfg_kind)[0]),
            "bc_aux_coef_end_declared": _aux_params(cfg_kind)[1],
            "bc_aux_anneal_fraction_declared": float(_aux_params(cfg_kind)[2]),
            "use_distributional_critic": False,
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
            # --- PRIMARY DV -----------------------------------------------------------------
            "competence_half_life_episodes_per_seed": [
                c.get("competence_half_life_episodes") for c in cells
            ],
            "competence_half_life_status_per_seed": statuses,
            "n_seeds_decayed": n_decayed,
            "n_seeds_did_not_decay": n_did_not_decay,
            "n_seeds_half_life_undefined": n_undefined,
            "half_life_effective_per_seed": effective,
            "half_life_effective_mean": eff_mean,
            "half_life_effective_censored_at": int(on_budget),
            # --- realised schedule (verification evidence) ------------------------------------
            "bc_aux_coef_first_per_seed": [c.get("bc_aux_coef_first") for c in cells],
            "bc_aux_coef_last_per_seed": [c.get("bc_aux_coef_last") for c in cells],
            "bc_aux_schedule_drop_per_seed": [c.get("bc_aux_schedule_drop") for c in cells],
            "bc_aux_schedule_drop_worst": round(drop_worst, 6),
            "bc_aux_schedule_drop_worst_seed": drop_worst_seed,
            "bc_aux_coef_first_worst": round(first_worst, 6),
            "bc_aux_coef_first_worst_seed": first_worst_seed,
            # --- shape + level ----------------------------------------------------------------
            "retained_fraction_per_seed": retained_vals,
            "retained_fraction_mean": retained_mean,
            "n_seeds_retained": n_retained,
            "retained_strict_majority": bool(n_cells and n_retained > (n_cells / 2.0)),
            "trajectory_peak_per_seed": [round(p, 6) for p in peaks],
            "trajectory_peak_mean": round(float(sum(peaks) / len(peaks)), 6) if peaks else 0.0,
            "n_seeds_peak_cleared_success_target": n_peak_cleared,
            "peak_cleared_strict_majority": bool(n_cells and n_peak_cleared > (n_cells / 2.0)),
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
        if ctx["is_annealed"]:
            measured["bc_aux_schedule_moved"] = float(r["bc_aux_schedule_drop_worst"])
            measured["bc_aux_schedule_starts_active"] = float(r["bc_aux_coef_first_worst"])
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, PRECONDITION_SPECS, measured=measured))
    gate = aggregate_arm_gates(arm_gates)

    # ---- routing: covariates + realised schedule + trajectory SHAPE first, level last ------
    const_r = per_arm_retention[BASELINE_ARM]
    anneal_r = per_arm_retention[ANNEALED_ARM]
    off_r = per_arm_retention[OFF_AUX_ARM]
    install_took_all = bool(
        const_r["install_took_strict_majority"]
        and anneal_r["install_took_strict_majority"]
        and off_r["install_took_strict_majority"]
    )
    schedule_verified = bool(
        anneal_r["bc_aux_schedule_drop_worst"] > BC_AUX_SCHEDULE_MOVE_FLOOR
        and anneal_r["bc_aux_coef_first_worst"] > BC_AUX_SCHEDULE_START_FLOOR
    )
    hl_const = const_r["half_life_effective_mean"]
    hl_anneal = anneal_r["half_life_effective_mean"]
    hl_off = off_r["half_life_effective_mean"]
    half_lives_defined = bool(
        hl_const is not None and hl_anneal is not None and hl_off is not None
        and const_r["n_seeds_half_life_undefined"] == 0
        and anneal_r["n_seeds_half_life_undefined"] == 0
        and off_r["n_seeds_half_life_undefined"] == 0
    )
    hl_margin_const_minus_off = (
        round(float(hl_const) - float(hl_off), 6) if half_lives_defined else None
    )
    hl_spread = (
        round(max(hl_const, hl_anneal, hl_off) - min(hl_const, hl_anneal, hl_off), 6)
        if half_lives_defined else None
    )
    ordered_by_persistence = bool(
        half_lives_defined and hl_const >= hl_anneal and hl_anneal >= hl_off
    )
    beats_off_by_margin = bool(
        hl_margin_const_minus_off is not None
        and hl_margin_const_minus_off >= half_life_margin
    )
    # The manipulation SUCCEEDED and then DECAYED, read on the most-persistent arm.
    succeeded_then_decayed = bool(
        const_r["peak_cleared_strict_majority"] and not const_r["retained_strict_majority"]
    )
    invariant_to_schedule = bool(
        half_lives_defined and hl_spread is not None and hl_spread < half_life_margin
        and not succeeded_then_decayed
    )
    c_load_bearing = bool(
        half_lives_defined and schedule_verified and ordered_by_persistence
        and beats_off_by_margin
    )

    if not anchors_ready or not install_took_all:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not gate["non_degenerate"]:
        outcome, label = "FAIL", "retention_auxiliary_grid_nondiscriminative"
    elif c_load_bearing:
        outcome, label = "PASS", "half_life_extended_by_persistence"
    elif succeeded_then_decayed:
        outcome, label = "FAIL", "retention_auxiliary_succeeded_then_decayed"
    elif invariant_to_schedule:
        outcome, label = "FAIL", "half_life_invariant_to_schedule"
    else:
        outcome, label = "FAIL", "retention_auxiliary_grid_nondiscriminative"

    degeneracy = check_degeneracy({
        "d3_boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        }
    })

    criteria_by_arm = {
        BASELINE_ARM: ["C_persistence_extends_competence_half_life"],
        ANNEALED_ARM: ["C_annealed_persistence_is_intermediate"],
        OFF_AUX_ARM: ["C_no_auxiliary_baseline_decays_fastest"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm, gate,
        extra={
            "C_persistence_extends_competence_half_life": bool(degeneracy["non_degenerate"]),
            "C_annealed_persistence_is_intermediate": bool(degeneracy["non_degenerate"]),
            "C_no_auxiliary_baseline_decays_fastest": bool(degeneracy["non_degenerate"]),
        },
    )
    criteria_nd["boot_arm_vs_anchor_foraging_spread"] = bool(degeneracy["non_degenerate"])
    criteria_nd["install_took_on_all_arms"] = install_took_all
    criteria_nd["annealed_schedule_verified_moved"] = schedule_verified
    criteria_nd["half_life_defined_on_all_arms"] = half_lives_defined
    criteria_nd["trajectory_has_shape_on_all_arms"] = bool(
        all(per_arm_retention[a]["n_trajectory_readings_worst"] > MIN_TRAJECTORY_READINGS
            for a in BOOT_ARMS)
    )

    interpretation = {
        "label": label,
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "criteria": [
            {"name": "C_persistence_extends_competence_half_life",
             "load_bearing": True,
             "description": (
                 "Auxiliary PERSISTENCE buys retention TIME: the effective competence half-life "
                 "is ordered constant >= annealed >= off, the annealed arm's realised schedule "
                 "is verified to have moved, and the constant arm beats the off arm by >= "
                 f"{HALF_LIFE_MARGIN_FRACTION_OF_BUDGET} of the RL budget "
                 f"({round(half_life_margin, 6)} episodes here)."
             ),
             "passed": c_load_bearing},
            {"name": "C_annealed_persistence_is_intermediate",
             "load_bearing": False,
             "description": (
                 "The annealed arm's effective half-life sits between the constant and off arms "
                 "-- the dose-response leg of the ordering, read only when its realised schedule "
                 "is verified."
             ),
             "passed": bool(
                 half_lives_defined and schedule_verified
                 and hl_const >= hl_anneal >= hl_off
             )},
            {"name": "C_no_auxiliary_baseline_decays_fastest",
             "load_bearing": False,
             "description": (
                 "The no-auxiliary arm does NOT hold the installed competence -- the contrast "
                 "the persistence arms are read against."
             ),
             "passed": bool(not off_r["retained_strict_majority"])},
        ],
        "criteria_non_degenerate": criteria_nd,
        "anchor_reachability": anchor_reachability,
        "interpretation_grid": [
            {"label": "substrate_not_ready_requeue", "outcome": "FAIL",
             "condition": (
                 "an anchor is sub-floor, OR post_bc_foraging_competence fails the 1.0 install "
                 "floor on a strict majority of seeds in any arm"
             ),
             "reading": (
                 "There is no installed prior to decay, so the half-life is UNDEFINED rather "
                 "than short and the run is UNINFORMATIVE about retention. NEVER a retention "
                 "verdict and never substrate_ceiling / substrate_conditional / "
                 "does_not_support / *_nondiscriminative / *_unmeetable. Requeue."
             )},
            {"label": "half_life_extended_by_persistence", "outcome": "PASS",
             "condition": (
                 "half-life defined on all three arms, the annealed arm's realised schedule "
                 "verified to have moved and to have started active, effective half-life "
                 "ordered constant >= annealed >= off, and constant beats off by the declared "
                 "margin"
             ),
             "reading": (
                 "THE HYPOTHESIS. The imitation auxiliary IS being out-competed by the RL "
                 "objective: how long the installed prior survives is set by how long the "
                 "auxiliary persists. The prior is not HELD, rather than not PROTECTED."
             )},
            {"label": "retention_auxiliary_succeeded_then_decayed", "outcome": "FAIL",
             "condition": (
                 "the most-persistent (constant) arm's trajectory PEAK cleared the "
                 "lift-competence target on a strict majority of seeds, but its terminal "
                 "retained_fraction fell below the retention floor"
             ),
             "reading": (
                 "The manipulation SUCCEEDED and then DECAYED -- a retention-DYNAMICS finding, "
                 "not a null. Read the trajectory and the half-life, not the terminal scalar. "
                 "This is the branch V3-EXQ-780's grid lacked, which is why a manipulation that "
                 "succeeded above target was scored a null there and self-routed "
                 "bc_prior_not_the_axis (rejected on autopsy)."
             )},
            {"label": "half_life_invariant_to_schedule", "outcome": "FAIL",
             "condition": (
                 "half-life defined on all three arms and their effective half-lives spread by "
                 "less than the declared margin"
             ),
             "reading": (
                 "THE DECLARED NULL. Half-life is invariant to the auxiliary schedule -> the "
                 "prior is NOT being out-competed by the RL objective. Removes "
                 "H-retention-auxiliary-decay from the live set; does NOT weaken MECH-457 "
                 "(diagnostic). Read jointly with H-retention-critic (V3-EXQ-788)."
             )},
            {"label": "retention_auxiliary_grid_nondiscriminative", "outcome": "FAIL",
             "condition": (
                 "no arm passed its precondition gate, OR a half-life is undefined where it "
                 "must be defined, OR the arms do not separate, OR the readouts fall in none of "
                 "the enumerated branches"
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
        "evidence_direction_per_claim": {"MECH-457": "unknown"},
        "per_arm_gate": gate["per_arm_gate"],
        "readiness": {
            "anchors_ready": anchors_ready,
            "install_took_all_arms": install_took_all,
            "annealed_schedule_verified": schedule_verified,
            "half_life_defined_all_arms": half_lives_defined,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
            "post_bc_worst_by_arm": {
                a: per_arm_retention[a]["post_bc_foraging_competence_worst"] for a in BOOT_ARMS
            },
            "post_bc_worst_seed_by_arm": {
                a: per_arm_retention[a]["post_bc_worst_seed"] for a in BOOT_ARMS
            },
            # P0 READINESS-ASSERT, restated flat and numerically so it is readable without
            # reconstructing the per-arm gate. SAME statistic the load-bearing criterion's
            # denominator uses (foraging_competence), measured on the WORST cell across all arms.
            "readiness_assert": {
                "name": "post_bc_install_took_worst_cell",
                "kind": "readiness",
                "description": (
                    "The BC install must have TAKEN before RL on every arm: post-BC / pre-RL "
                    "foraging_competence of the worst cell clears the 1.0 install floor. Below "
                    "it there is no installed prior to decay and the run self-routes "
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
            # The SCHEDULE-VERIFICATION assert, restated flat for the same reason. A two-sided
            # pin: the annealed schedule must MOVE (else it aliases onto the constant arm) and
            # must START ACTIVE (else it aliases onto the off arm).
            "schedule_assert": {
                "name": "annealed_bc_aux_schedule_moved_and_started_active",
                "kind": "manipulation_active",
                "description": (
                    "The annealed arm's REALISED auxiliary weight, from train_a2c's "
                    "bc_aux_coef_first / bc_aux_coef_last on its worst cell. drop > "
                    f"{BC_AUX_SCHEDULE_MOVE_FLOOR} and first > {BC_AUX_SCHEDULE_START_FLOOR}."
                ),
                "control": "annealed arm worst-cell realised bc_aux_coef_first / _last",
                "direction": "lower",
                "measured_drop": float(anneal_r["bc_aux_schedule_drop_worst"]),
                "threshold_drop": float(BC_AUX_SCHEDULE_MOVE_FLOOR),
                "measured_first": float(anneal_r["bc_aux_coef_first_worst"]),
                "threshold_first": float(BC_AUX_SCHEDULE_START_FLOOR),
                "declared_drop_over_budget": round(declared_drop, 6),
                "met": schedule_verified,
            },
        },
        "headline": {
            "persistence_extends_half_life": c_load_bearing,
            "succeeded_then_decayed": succeeded_then_decayed,
            "half_life_invariant_to_schedule": invariant_to_schedule,
            "half_life_effective_mean_constant": hl_const,
            "half_life_effective_mean_annealed": hl_anneal,
            "half_life_effective_mean_off": hl_off,
            "half_life_margin_constant_minus_off": hl_margin_const_minus_off,
            "half_life_spread_across_arms": hl_spread,
            "half_life_margin_episodes": round(half_life_margin, 6),
            "half_life_margin_fraction_of_budget": HALF_LIFE_MARGIN_FRACTION_OF_BUDGET,
            "half_life_ordered_by_persistence": ordered_by_persistence,
            "n_did_not_decay_by_arm": {
                a: per_arm_retention[a]["n_seeds_did_not_decay"] for a in BOOT_ARMS
            },
            "n_half_life_undefined_by_arm": {
                a: per_arm_retention[a]["n_seeds_half_life_undefined"] for a in BOOT_ARMS
            },
            "retained_fraction_mean_by_arm": {
                a: per_arm_retention[a]["retained_fraction_mean"] for a in BOOT_ARMS
            },
            "bc_aux_schedule_drop_worst_annealed": anneal_r["bc_aux_schedule_drop_worst"],
            "bc_aux_coef_first_worst_annealed": anneal_r["bc_aux_coef_first_worst"],
            "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
            "peak_success_target": PEAK_SUCCESS_TARGET,
            "retention_probe_every": int(probe_every),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
        },
        "per_arm_retention": per_arm_retention,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "post_bc_install_floor": float(POST_BC_INSTALL_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
            "post_bc_780_reference_raw_view": 20.933,
            "terminal_780_reference_raw_view": 11.667,
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
        "headline": result["headline"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_retention": result["per_arm_retention"],
        "reference_band": result["reference_band"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": [BASELINE_ARM],
            "reuse_eligible": True,
            "note": (
                "The shared baseline arm (" + BASELINE_ARM + ", the 780 condition: constant "
                "bc_aux_coef=0.5, scalar critic) is emitted reuse-ELIGIBLE with the SHARED "
                "lineage slice (experiments/_lib/baselines/mech457_retention.off_path_config_"
                "slice) and include_driver_script_in_hash=False, so it interchanges with the "
                "cell the sibling retention leg V3-EXQ-788 (H-retention-critic) mints for its "
                "scalar control across the two DIFFERENT drivers. The flag matches on both "
                "sides, which is what makes the reuse possible at all. First leg to run mints; "
                "second reuses. No separate baseline-only mint job (neither sanctioned "
                "exception applies)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The competence HALF-LIFE -- episodes until post-installation competence first "
            "falls below half the installed (post-BC) value -- computed from the mid-training "
            f"competence TRAJECTORY probed every {cfg['retention_probe_every']} episodes, for a "
            "BC-installed raw_view policy under three auxiliary PERSISTENCE schedules (constant "
            "0.5 / annealed 0.5->0.0 via linear_anneal / off 0.0). PASS: half-life defined on "
            "all three arms, the annealed arm's realised schedule verified to have moved and to "
            "have started active, effective half-life ordered constant >= annealed >= off, and "
            f"constant beating off by >= {HALF_LIFE_MARGIN_FRACTION_OF_BUDGET} of the RL budget. "
            "A half-life of None is BRANCHED, never coerced: 'undefined_no_install' (install "
            "~0 -- the ratio does not exist) and 'did_not_decay' (never halved -- right-censored "
            "at the budget) are recorded as distinct statuses. Readiness: "
            "post_bc_foraging_competence clears the 1.0 install floor on the worst seed of each "
            "arm (780 measured 20.933 on raw_view, 3/3 taking); an install that did not take "
            "self-routes substrate_not_ready_requeue."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 RETENTION leg H-retention-auxiliary-decay, pre-registered "
            "under question 'competence_floor' in hypothesis_space_registry.v1.json. DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy (read "
            "jointly with H-retention-critic, V3-EXQ-788). MANIPULATION = the imitation "
            "auxiliary's PERSISTENCE ONLY (the bc_aux_coef / bc_aux_coef_end / "
            "bc_aux_anneal_fraction triple). The anneal uses linear_anneal, NOT warm_then_anneal "
            "-- the latter is parameterised by the SHARED warm_start_fraction and would couple "
            "BC persistence to the exploration anneal, confounding the single intervention. "
            "ANTI-ALIAS: use_distributional_critic stays False on ALL arms (that is 788's "
            "locus). MOTIVATION: V3-EXQ-780 ran a CONSTANT bc_aux_coef=0.5 throughout and still "
            "lost 20.933 -> 11.667, so a persistent-but-fixed auxiliary is insufficient and the "
            "SCHEDULE dimension is untested. raw_view ONLY (780: post-BC 20.933 with 3/3 seeds "
            "taking on raw_view vs 0.583 with 0/3 on z_world). Reference build 128-wide / 3x "
            "budget / z_world detached / credit-replay 3 / topk 32 via "
            "baselines.reference_config -- NOT the 769-falsified 256/5x regression. SCHEDULE "
            "VERIFICATION is a numeric precondition scoped to the annealed arm (realised "
            "bc_aux_coef_first/_last), because the schedule drives the auxiliary's GUARD as well "
            "as its weight: a flat schedule would alias onto the constant arm and a collapsed "
            "one onto the off arm. DECLARED NULL: half-life is invariant to the auxiliary "
            "schedule -> the prior is NOT being out-competed by the RL objective (label "
            "half_life_invariant_to_schedule); this does NOT weaken MECH-457, which stays "
            "candidate/v3_pending."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-789 MECH-457 GOV-FANOUT-1 H-retention-auxiliary-decay"
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
        "bc_aux_coef_start": float(BC_AUX_COEF_START),
        "bc_aux_coef_anneal_end": float(BC_AUX_COEF_ANNEAL_END),
        "bc_aux_anneal_fraction": float(BC_AUX_ANNEAL_FRACTION),
        "bc_aux_coef_off": float(BC_AUX_COEF_OFF),
        "bc_aux_schedule_fn": "linear_anneal (NOT warm_then_anneal -- see module docstring)",
        "bc_demonstrator": "local_view_greedy",
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": baselines.REF_CREDIT_PASSES,
        "ref_credit_topk": baselines.REF_CREDIT_TOPK,
        "ref_cotrain_encoder": baselines.REF_COTRAIN_ENCODER,
        "use_distributional_critic_all_arms": False,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA, "bc_lr": fan.BC_LR,
        "constant_config": _make_cfg("constant", on_budget, probe_every).as_slice(),
        "annealed_config": _make_cfg("annealed", on_budget, probe_every).as_slice(),
        "off_config": _make_cfg("off", on_budget, probe_every).as_slice(),
        "declared_annealed_schedule_drop": round(_declared_schedule_drop(on_budget), 6),
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "half_life_margin_fraction_of_budget": HALF_LIFE_MARGIN_FRACTION_OF_BUDGET,
        "half_life_margin_episodes": round(
            float(HALF_LIFE_MARGIN_FRACTION_OF_BUDGET) * float(on_budget), 6
        ),
        "bc_aux_schedule_move_floor": BC_AUX_SCHEDULE_MOVE_FLOOR,
        "bc_aux_schedule_start_floor": BC_AUX_SCHEDULE_START_FLOOR,
        "peak_success_target": PEAK_SUCCESS_TARGET,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 retention (H-retention-auxiliary-decay)",
        "hypothesis_id": "H-retention-auxiliary-decay",
        "hypothesis_question": "competence_floor",
        "sibling_leg": "V3-EXQ-788 (H-retention-critic)",
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
        f"schedule_verified={result['readiness']['annealed_schedule_verified']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for arm_id in BOOT_ARMS:
        r = result["per_arm_retention"][arm_id]
        print(
            f"  {arm_id}: post_bc_worst={r['post_bc_foraging_competence_worst']} "
            f"(seed={r['post_bc_worst_seed']}) install_took={r['n_seeds_install_took']}/"
            f"{r['n_cells']} readings_worst={r['n_trajectory_readings_worst']} "
            f"half_life_eff_mean={r['half_life_effective_mean']} "
            f"(decayed={r['n_seeds_decayed']} did_not_decay={r['n_seeds_did_not_decay']} "
            f"undefined={r['n_seeds_half_life_undefined']}) "
            f"aux_first={r['bc_aux_coef_first_worst']} drop={r['bc_aux_schedule_drop_worst']} "
            f"retained_frac_mean={r['retained_fraction_mean']} "
            f"terminal={r['terminal_forage_mean']}", flush=True,
        )
    print(
        f"  margin(constant-off)={hl['half_life_margin_constant_minus_off']} "
        f"spread={hl['half_life_spread_across_arms']} "
        f"margin_floor={hl['half_life_margin_episodes']} "
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
