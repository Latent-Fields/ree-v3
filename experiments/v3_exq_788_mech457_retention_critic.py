#!/opt/local/bin/python3
"""V3-EXQ-788 -- MECH-457 GOV-FANOUT-1 RETENTION leg, H-retention-critic: does a better VALUE
ESTIMATOR let an installed competent policy be REFINED rather than eroded?

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PRE-REGISTERED as hypothesis H-retention-critic under question qid "competence_floor" in
REE_assembly/evidence/planning/hypothesis_space_registry.v1.json.

THE HYPOTHESIS. A flat / uninformed value baseline fails to refine an installed competent
policy. V3-EXQ-780 showed the BC install DOES take on raw_view (post-BC foraging competence
20.933, 3/3 seeds clearing the floor; against 0.583 with 0/3 taking on z_world -- which is why
this leg is raw_view ONLY: a retention question is unanswerable where the install never took).
V3-EXQ-782 R-(b) then measured the shared CTRL critic FLAT AND UNINFORMED: std(V)/std(G) =
0.041 against a 0.25 collapse threshold, and a pre-reward-vs-far separation ratio of 0.016
against a 0.25 floor. A policy-gradient update weighted by such a baseline is close to
unweighted, so the installed prior is not refined -- it is washed out.

DESIGN. BC-install the raw_view policy to the ~20.9 competence point, then run the SAME RL
refinement under two value estimators:
  * retcritic_scalar          (OFF / control) -- the current scalar-MSE critic.
  * retcritic_distributional  (ON  / treat)   -- a categorical (two-hot / HL-Gauss) critic over
                                                a symlog bin support, decoded by expectation.
ANTI-ALIAS (load-bearing). ONLY the value estimator changes. The update rule is untouched:
policy loss, advantage weighting, entropy bonus, credit replay and the BC auxiliary are
byte-identical across the two arms (see mech457_fanout.critic_value_loss, which dispatches the
VALUE term alone). The update-constraint locus belongs to the sibling hypothesis
H-retention-consolidation; a leg that moved both would read as neither.

BOTH ARMS ARE BUILT THROUGH experiments/_lib/baselines/mech457_retention.py, so this leg's OFF
cell and V3-EXQ-789's OFF cell agree by CONSTRUCTION rather than by two drivers happening to
spell the same config. Everything outside the single declared manipulation -- bc_aux_coef,
capacity (128-wide), budget (3x), credit replay 3/topk 32, the developmental drive anneal, the
install -- is supplied by baselines.reference_config() and is IDENTICAL across the two arms.
The 769-FALSIFIED capacity regression (boot.make_on_config / ON_ACTOR_CRITIC_HIDDEN=256 /
ON_BUDGET_MULTIPLIER=5) is deliberately NOT used anywhere in this driver.

THE DV IS A TRAJECTORY, NOT A TERMINAL SCALAR. cfg.retention_probe_every wires the substrate's
non-perturbing mid-training competence probe (train_a2c snapshots and restores the torch/numpy/
random streams around every reading, so measurement neutrality is a substrate guarantee). At
250-episode cadence over the 3000-episode budget that is 12 readings per (arm x seed), each
recorded in full in the manifest. Terminal-only measurement is what kept this deficit invisible
for ten legs; collapsing the trajectory here would reproduce that defect.

INTERPRETATION GRID (five enumerated branches; the "manipulation succeeded and then decayed"
branch is mandatory and is the one V3-EXQ-780 lacked):
  * substrate_not_ready_requeue            -- the install did NOT take (or an anchor is
      sub-floor). UNINFORMATIVE about retention. NEVER a retention verdict, and never
      substrate_ceiling / substrate_conditional / does_not_support.
  * retention_critic_retains_competence    -- PASS. The distributional arm holds the installed
      competence (retained fraction >= floor) and beats the scalar arm by the declared margin.
  * retention_critic_succeeded_then_decayed-- the distributional arm ROSE to at/above the
      lift-competence target at its trajectory PEAK and then FELL below the retention floor.
      The manipulation worked and then decayed: a retention-DYNAMICS finding, not a null.
  * retention_eroded_under_both            -- THE DECLARED NULL. The installed prior erodes
      identically under both critics -> the critic baseline is NOT the retention mechanism.
  * retention_grid_nondiscriminative       -- no arm passed its gate, or the arms/anchors do
      not separate. Not a refutation.

ROUTING CONSUMES THE DECLARED COVARIATES, NOT ONLY THE TERMINAL CRITERION.
post_bc_foraging_competence is LOAD-BEARING in the routing (it selects the requeue branch
before any retention branch can be reached), and the trajectory SHAPE (peak vs terminal)
selects between the retained / succeeded-then-decayed / eroded branches. This is the exact
defect that sank V3-EXQ-780: post_bc_foraging_competence was present and declared there, but
its grid enumerated only a ~0 null, so a manipulation that succeeded ABOVE target was scored a
null and self-routed bc_prior_not_the_axis (rejected on autopsy).

MULTI-ARM GATE. experiments/_lib/precondition_gate.py -- per-arm gates aggregated with
aggregate_arm_gates so non_degenerate = ANY arm green. One arm's unmet precondition must NEVER
vacate the other's valid result (the V3-EXQ-785 defect).

MINT (mint-as-you-go). The OFF arm emits its cell reuse-ELIGIBLE with the shared lineage slice
(baselines.off_path_config_slice) and include_driver_script_in_hash=False, so the sibling leg
V3-EXQ-789 can reuse the minted OFF cell across its different driver. The first of the two legs
to run mints; the second reuses. No separate baseline-only mint job.

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
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_788_mech457_retention_critic"
QUEUE_ID = "V3-EXQ-788"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# --- Probe cadence (the measurement constraint the whole leg exists to satisfy) ------------
# 3000-episode reference budget / 250 = 12 readings per (arm x seed). Declared as a module
# constant (not derived at call time) because it is fingerprint-declared in the config_slice:
# a probed and an unprobed cell are bit-identical COMPUTATIONS but are not interchangeable
# ARTIFACTS, since only one carries the trajectory a consumer reads.
RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# --- Pre-registered retention thresholds (declared; never derived from the run) ------------
# retained_fraction = terminal trajectory competence / installed (post-BC) competence.
RETAINED_FRACTION_FLOOR = 0.5          # >= half the installed competence survives refinement
RETENTION_ARM_MARGIN = 0.15            # distributional must beat scalar by this in retained frac
PEAK_SUCCESS_TARGET = baselines.LIFT_COMPETENCE_TARGET   # ~13.05 res/ep
MIN_TRAJECTORY_READINGS = 1.5          # need >= 2 readings for a SHAPE (floor, strict >)

# The install / reference build. Re-exported from the shared lineage module so this driver
# cannot drift from it.
REF_REPRESENTATION = baselines.REF_REPRESENTATION          # raw_view ONLY
REF_ACTOR_CRITIC_HIDDEN = baselines.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = baselines.REF_BUDGET_MULTIPLIER      # 3x
POST_BC_INSTALL_FLOOR = baselines.POST_BC_INSTALL_FLOOR      # 1.0 competence floor

CFG_KINDS: Tuple[str, ...] = ("scalar", "distributional")
OFF_KIND = "scalar"


def _arm_id(cfg_kind: str) -> str:
    return f"retcritic_{cfg_kind}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(k) for k in CFG_KINDS)
OFF_ARM = _arm_id(OFF_KIND)
ON_ARM = _arm_id("distributional")
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


# ---------------------------------------------------------------------------------------
# Precondition specs (multi-arm gate). Every entry carries numeric measured + threshold so
# build_experiment_indexes.py can recompute `met`; all four are FLOORS (direction defaults to
# "lower"), so none needs the "upper" ceiling tag.
# ---------------------------------------------------------------------------------------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="local_view_greedy_clears_floor_at_d3",
        description=(
            "LocalViewGreedyPolicy reading the SAME 5x5 resource_field_view forages above the "
            "1.0 competence floor at D3 -- the positive control that the env is solvable from "
            "the local view. Below-floor means the substrate/env is not ready, NOT that the "
            "critic failed."
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
        name="distributional_value_head_enabled",
        description=(
            "The treatment arm must actually be running the categorical value head (1 = on). "
            "Scoped OUT of the scalar control arm, where a distributional head is not merely "
            "absent but is the very thing being controlled against -- asserting it there would "
            "make the control's gate structurally un-passable and collapse the two-arm design."
        ),
        control="cfg.use_distributional_critic as constructed for this arm",
        threshold=0.5,
        kind="manipulation_active",
        applies_to=lambda ctx: bool(ctx["use_distributional_critic"]),
        applies_note=(
            "treatment arm only -- the scalar CONTROL is defined by the absence of the "
            "distributional head, so this precondition is not meaningful for it"
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
    """The shared reference RL-refinement config, varying ONLY use_distributional_critic.

    Everything else -- bc_aux_coef, actor_critic_hidden, budget, credit replay, the drive
    anneal, the install -- comes from baselines.reference_config() and is therefore IDENTICAL
    across the two arms by construction. This is the leg's anti-alias: the update rule is
    untouched; only what the baseline knows differs.
    """
    if cfg_kind not in CFG_KINDS:
        raise ValueError(f"unknown cfg_kind {cfg_kind!r}")
    return baselines.reference_config(
        on_budget,
        use_distributional_critic=(cfg_kind == "distributional"),
        retention_probe_every=int(probe_every),
    )


def _config_slice(cfg_kind: str, env_kwargs: Dict[str, Any], eval_eps: int, steps: int,
                  on_budget: int, probe_every: int) -> Dict[str, Any]:
    """Both arms declare the SHARED lineage slice; the ON arm then overrides its own arm_id and
    folds in its (distributional) config. Anchoring both on off_path_config_slice makes the
    OFF-cell match with V3-EXQ-789 structural rather than accidental."""
    base = baselines.off_path_config_slice(
        env_kwargs, eval_eps=int(eval_eps), steps=int(steps),
        on_budget=int(on_budget), retention_probe_every=int(probe_every),
    )
    if cfg_kind == OFF_KIND:
        return base
    base["arm_id"] = _arm_id(cfg_kind)
    base["kind"] = "mech457_retention_critic_treatment"
    base.update(_make_cfg(cfg_kind, on_budget, probe_every).as_slice())
    return base


def _build_rep(cfg_kind: str, seed: int, env_kwargs: Dict[str, Any], steps: int, cfg) -> Any:
    """Construct the raw_view RepAgent for one arm.

    DEVIATION, recorded deliberately. The OFF arm goes through
    baselines.build_off_arm(), which is the mint path and must stay exact. That helper does
    NOT forward use_distributional_critic to mech.make_rep (the critic swap is applied at REP
    CONSTRUCTION), so the treatment arm cannot be built through it without silently collapsing
    into the control. The treatment branch therefore calls mech.make_rep directly with the SAME
    arguments build_off_arm passes, all read from the SAME cfg, plus the one declared
    manipulation.
    """
    if cfg_kind == OFF_KIND:
        return baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)
    warm_env = x734._make_env(seed, env_kwargs)
    return mech.make_rep(
        baselines.REF_REPRESENTATION, warm_env, seed=seed, p0=0, steps=int(steps),
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
        use_distributional_critic=bool(cfg.use_distributional_critic),
    )


def _run_boot_cell(cfg_kind: str, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind)
    cfg = _make_cfg(cfg_kind, on_budget, probe_every)
    rep_agent = _build_rep(cfg_kind, seed, env_kwargs, steps, cfg)

    # Phase 1 -- BC install to the ~20.9 competence point, then measure whether it TOOK.
    install = baselines.install_bc_prior(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    post_bc = float(install["post_bc_foraging_competence"])

    # Phase 2 -- the SAME RL refinement under this arm's value estimator, probed on a cadence.
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

    row["post_bc_foraging_competence"] = round(post_bc, 6)
    row["install_took"] = bool(install["install_took"])
    row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
    row["use_distributional_critic"] = bool(cfg.use_distributional_critic)
    # FULL per-seed trajectory -- never collapsed to a terminal scalar.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    row["retained_fraction"] = retained
    row["competence_half_life_episodes"] = half_life
    row["peak_cleared_success_target"] = bool(peak >= PEAK_SUCCESS_TARGET)
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
            "use_distributional_critic": (k == "distributional"),
            "n_episodes": int(on_budget),
            "probe_every": int(probe_every),
        }
        for k in CFG_KINDS
    ]


def run_experiment(seeds: List[int], on_budget: int, eval_eps: int, steps: int,
                   probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-retention-critic "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep={REF_REPRESENTATION}, ON_budget={on_budget}, probe_every={probe_every}, "
        f"eval={eval_eps}, steps={steps}; ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, "
        f"budget_mult={REF_BUDGET_MULTIPLIER}; manipulation=use_distributional_critic ONLY)",
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
        # OFF arm mints cross-driver-reusable (no driver script in the hash) so V3-EXQ-789 can
        # reuse it; the treatment arm is leg-specific and keeps the driver in its hash.
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
        retained_vals = [c.get("retained_fraction") for c in cells]
        retained_num = [float(v) for v in retained_vals if v is not None]
        retained_mean = round(float(sum(retained_num) / len(retained_num)), 6) if retained_num else None
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        n_peak_cleared = int(sum(1 for c in cells if bool(c.get("peak_cleared_success_target", False))))
        n_retained = int(sum(1 for v in retained_num if v >= RETAINED_FRACTION_FLOOR))
        n_cells = len(cells)
        per_arm_retention[arm_id] = {
            "arm_id": arm_id,
            "use_distributional_critic": (cfg_kind == "distributional"),
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
        if ctx["use_distributional_critic"]:
            measured["distributional_value_head_enabled"] = 1.0
        arm_gates.append(evaluate_arm_gate(arm_id, ctx, PRECONDITION_SPECS, measured=measured))
    gate = aggregate_arm_gates(arm_gates)

    # ---- routing: covariates + trajectory SHAPE first, terminal criterion last -------------
    off_r = per_arm_retention[OFF_ARM]
    on_r = per_arm_retention[ON_ARM]
    install_took_both = bool(
        off_r["install_took_strict_majority"] and on_r["install_took_strict_majority"]
    )
    on_retained = bool(on_r["retained_strict_majority"])
    off_retained = bool(off_r["retained_strict_majority"])
    on_frac = on_r["retained_fraction_mean"]
    off_frac = off_r["retained_fraction_mean"]
    retained_margin = (
        round(float(on_frac) - float(off_frac), 6)
        if (on_frac is not None and off_frac is not None) else None
    )
    beats_scalar_by_margin = bool(
        retained_margin is not None and retained_margin >= RETENTION_ARM_MARGIN
    )
    on_succeeded_then_decayed = bool(on_r["peak_cleared_strict_majority"] and not on_retained)
    eroded_under_both = bool(
        (not on_retained) and (not off_retained)
        and (retained_margin is None or abs(retained_margin) < RETENTION_ARM_MARGIN)
        and not on_succeeded_then_decayed
    )
    c_load_bearing = bool(on_retained and beats_scalar_by_margin)

    if not anchors_ready or not install_took_both:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not gate["non_degenerate"]:
        outcome, label = "FAIL", "retention_grid_nondiscriminative"
    elif c_load_bearing:
        outcome, label = "PASS", "retention_critic_retains_competence"
    elif on_succeeded_then_decayed:
        outcome, label = "FAIL", "retention_critic_succeeded_then_decayed"
    elif eroded_under_both:
        outcome, label = "FAIL", "retention_eroded_under_both"
    else:
        outcome, label = "FAIL", "retention_grid_nondiscriminative"

    degeneracy = check_degeneracy({
        "d3_boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        }
    })

    criteria_by_arm = {
        ON_ARM: ["C_distributional_critic_retains_installed_competence"],
        OFF_ARM: ["C_scalar_critic_baseline_erodes_installed_competence"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm, gate,
        extra={
            "C_distributional_critic_retains_installed_competence": bool(
                degeneracy["non_degenerate"]
            ),
            "C_scalar_critic_baseline_erodes_installed_competence": bool(
                degeneracy["non_degenerate"]
            ),
        },
    )
    criteria_nd["boot_arm_vs_anchor_foraging_spread"] = bool(degeneracy["non_degenerate"])
    criteria_nd["install_took_on_both_arms"] = install_took_both
    criteria_nd["trajectory_has_shape_on_both_arms"] = bool(
        off_r["n_trajectory_readings_worst"] > MIN_TRAJECTORY_READINGS
        and on_r["n_trajectory_readings_worst"] > MIN_TRAJECTORY_READINGS
    )

    interpretation = {
        "label": label,
        "preconditions": gate["adjudication_preconditions"],
        "preconditions_scope_note": gate["per_arm_gate"]["preconditions_scope_note"],
        "criteria": [
            {"name": "C_distributional_critic_retains_installed_competence",
             "load_bearing": True,
             "description": (
                 "The distributional-critic arm holds a strict majority of seeds at "
                 f"retained_fraction >= {RETAINED_FRACTION_FLOOR} AND beats the scalar arm's "
                 f"mean retained_fraction by >= {RETENTION_ARM_MARGIN}."
             ),
             "passed": c_load_bearing},
            {"name": "C_scalar_critic_baseline_erodes_installed_competence",
             "load_bearing": False,
             "description": (
                 "The scalar control arm does NOT hold the installed competence -- the "
                 "contrast the treatment is read against."
             ),
             "passed": bool(not off_retained)},
        ],
        "criteria_non_degenerate": criteria_nd,
        "anchor_reachability": anchor_reachability,
        "interpretation_grid": [
            {"label": "substrate_not_ready_requeue", "outcome": "FAIL",
             "condition": (
                 "an anchor is sub-floor, OR post_bc_foraging_competence fails the 1.0 install "
                 "floor on a strict majority of seeds in either arm"
             ),
             "reading": (
                 "There is no installed prior to retain, so the run is UNINFORMATIVE about "
                 "retention. NEVER a retention verdict and never substrate_ceiling / "
                 "substrate_conditional / does_not_support. Requeue."
             )},
            {"label": "retention_critic_retains_competence", "outcome": "PASS",
             "condition": (
                 "distributional arm retains on a strict majority of seeds AND beats the "
                 "scalar arm's mean retained_fraction by the declared margin"
             ),
             "reading": (
                 "A flat/uninformed value baseline WAS the retention deficit: an informed "
                 "critic lets the installed policy be refined rather than washed out."
             )},
            {"label": "retention_critic_succeeded_then_decayed", "outcome": "FAIL",
             "condition": (
                 "distributional arm's trajectory PEAK cleared the lift-competence target on a "
                 "strict majority of seeds, but its terminal retained_fraction fell below the "
                 "retention floor"
             ),
             "reading": (
                 "The manipulation SUCCEEDED and then DECAYED -- a retention-DYNAMICS finding, "
                 "not a null. Read the trajectory and the half-life, not the terminal scalar. "
                 "This is the branch V3-EXQ-780's grid lacked, which is why a manipulation that "
                 "succeeded above target was scored a null there."
             )},
            {"label": "retention_eroded_under_both", "outcome": "FAIL",
             "condition": (
                 "neither arm retains and their mean retained_fraction differ by less than the "
                 "declared margin"
             ),
             "reading": (
                 "THE DECLARED NULL. The installed prior erodes identically under both critics "
                 "-> the critic baseline is NOT the retention mechanism. Removes "
                 "H-retention-critic from the live set; does NOT weaken MECH-457 (diagnostic). "
                 "Read jointly with H-retention-auxiliary-decay (V3-EXQ-789)."
             )},
            {"label": "retention_grid_nondiscriminative", "outcome": "FAIL",
             "condition": (
                 "no arm passed its precondition gate, or the arms/anchors do not separate, or "
                 "the readouts fall in none of the enumerated branches"
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
            "install_took_both_arms": install_took_both,
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
            # on (foraging_competence), measured on the WORST cell across both arms.
            "readiness_assert": {
                "name": "post_bc_install_took_worst_cell",
                "kind": "readiness",
                "description": (
                    "The BC install must have TAKEN before RL on every arm: post-BC / pre-RL "
                    "foraging_competence of the worst cell clears the 1.0 install floor. Below "
                    "it there is no installed prior to retain and the run self-routes "
                    "substrate_not_ready_requeue."
                ),
                "control": "post_bc_foraging_competence, worst cell over both arms x all seeds",
                "direction": "lower",
                "measured": round(min(
                    float(per_arm_retention[a]["post_bc_foraging_competence_worst"])
                    for a in BOOT_ARMS
                ), 6) if all(per_arm_retention[a]["n_cells"] for a in BOOT_ARMS) else 0.0,
                "threshold": float(POST_BC_INSTALL_FLOOR),
                "met": install_took_both,
            },
        },
        "headline": {
            "distributional_retains_installed_competence": c_load_bearing,
            "distributional_succeeded_then_decayed": on_succeeded_then_decayed,
            "eroded_under_both": eroded_under_both,
            "retained_fraction_mean_distributional": on_frac,
            "retained_fraction_mean_scalar": off_frac,
            "retained_fraction_margin_on_minus_off": retained_margin,
            "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
            "retention_arm_margin": RETENTION_ARM_MARGIN,
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
            "reusable_arms": [OFF_ARM],
            "reuse_eligible": True,
            "note": (
                "The OFF arm (" + OFF_ARM + ") is emitted reuse-ELIGIBLE with the SHARED "
                "lineage slice (experiments/_lib/baselines/mech457_retention.off_path_config_"
                "slice) and include_driver_script_in_hash=False, so the sibling retention leg "
                "V3-EXQ-789 (H-retention-auxiliary-decay) can reuse the minted cell across its "
                "different driver. First leg to run mints; second reuses. No separate "
                "baseline-only mint job (neither sanctioned exception applies)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The post-installation competence TRAJECTORY (not terminal competence) of a "
            "BC-installed raw_view policy across the reference RL refinement, probed every "
            f"{cfg['retention_probe_every']} episodes, under a DISTRIBUTIONAL vs a SCALAR "
            "critic. Statistic: retained_fraction = terminal trajectory competence / installed "
            f"(post-BC) competence, with the trajectory PEAK read alongside it. PASS: the "
            f"distributional arm holds retained_fraction >= {RETAINED_FRACTION_FLOOR} on a "
            f"strict majority of seeds AND beats the scalar arm's mean by >= "
            f"{RETENTION_ARM_MARGIN}. Readiness: post_bc_foraging_competence clears the 1.0 "
            "install floor on the worst seed of each arm (780 measured 20.933 on raw_view, "
            "3/3 taking); an install that did not take self-routes substrate_not_ready_requeue."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 RETENTION leg H-retention-critic, pre-registered under "
            "question 'competence_floor' in hypothesis_space_registry.v1.json. DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy (read "
            "jointly with H-retention-auxiliary-decay, V3-EXQ-789). MANIPULATION = the VALUE "
            "ESTIMATOR ONLY (use_distributional_critic); the update rule is untouched -- policy "
            "loss, advantage weighting, entropy bonus, credit replay and the BC auxiliary are "
            "identical across arms. That anti-alias is load-bearing: the update-constraint "
            "locus belongs to the sibling hypothesis H-retention-consolidation. MOTIVATION: "
            "V3-EXQ-782 R-(b) measured the shared CTRL critic flat and uninformed (std(V)/"
            "std(G)=0.041 vs a 0.25 collapse threshold; pre-reward-vs-far separation 0.016 vs a "
            "0.25 floor). raw_view ONLY (780: post-BC 20.933 with 3/3 seeds taking on raw_view "
            "vs 0.583 with 0/3 on z_world). Reference build 128-wide / 3x budget / z_world "
            "detached / credit-replay 3 / topk 32 via baselines.reference_config -- NOT the "
            "769-falsified 256/5x regression. DECLARED NULL: the installed prior erodes "
            "identically under both critics -> the critic baseline is NOT the retention "
            "mechanism (label retention_eroded_under_both); this does NOT weaken MECH-457, "
            "which stays candidate/v3_pending."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-788 MECH-457 GOV-FANOUT-1 H-retention-critic"
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
        "scalar_config": _make_cfg("scalar", on_budget, probe_every).as_slice(),
        "distributional_config": _make_cfg("distributional", on_budget, probe_every).as_slice(),
        "dist_critic_n_bins": fan.DIST_CRITIC_N_BINS,
        "dist_critic_limit": fan.DIST_CRITIC_LIMIT,
        "dist_critic_sigma": fan.DIST_CRITIC_SIGMA,
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "retention_arm_margin": RETENTION_ARM_MARGIN,
        "peak_success_target": PEAK_SUCCESS_TARGET,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 retention (H-retention-critic)",
        "hypothesis_id": "H-retention-critic",
        "hypothesis_question": "competence_floor",
        "sibling_leg": "V3-EXQ-789 (H-retention-auxiliary-decay)",
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
        f"install_took_both={result['readiness']['install_took_both_arms']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    for arm_id in BOOT_ARMS:
        r = result["per_arm_retention"][arm_id]
        print(
            f"  {arm_id}: post_bc_worst={r['post_bc_foraging_competence_worst']} "
            f"(seed={r['post_bc_worst_seed']}) install_took={r['n_seeds_install_took']}/"
            f"{r['n_cells']} readings_worst={r['n_trajectory_readings_worst']} "
            f"retained_frac_mean={r['retained_fraction_mean']} "
            f"peak_mean={r['trajectory_peak_mean']} "
            f"terminal={r['terminal_forage_mean']}", flush=True,
        )
    print(
        f"  margin(on-off)={hl['retained_fraction_margin_on_minus_off']} "
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
