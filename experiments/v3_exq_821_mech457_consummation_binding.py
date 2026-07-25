#!/opt/local/bin/python3
"""V3-EXQ-821 -- MECH-457 GOV-FANOUT-1 RETENTION leg 4, H-consummation-binding: was V3-EXQ-781's
drive-side NULL an artefact of a MISSING CONSUMMATORY ACT?

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PRE-REGISTERED as hypothesis H-consummation-binding under question qid "competence_floor" in
REE_assembly/evidence/planning/hypothesis_space_registry.v1.json (mech457_retention_portfolio
2026-07-18, leg 4; competence_floor_reposing 2026-07-25).

THE HYPOTHESIS. V3-EXQ-781 gave the composed bootstrap an innate, NON-EXTINGUISHING appetitive
approach drive and found the drive EARNED (0.707) while raw-view foraging was SUPPRESSED to 0.200
from a 2.983 control -- approach WITHOUT consummation. The hypothesis is that 781's drive-side
null was itself an artefact of a MISSING CONSUMMATORY ACT: its terminal drive was
non-extinguishing, so there was no mechanism for the approach drive to terminate correctly on
arrival and hand off to a distinct act of consuming. The mech457_consummatory_act env node
(2026-07-25) makes contact AFFORD rather than EFFECT consumption (a distinct CONSUME action), and
mech457_approach_extinction (2026-07-25) lets the approach drive EXTINGUISH on contact
(approach_extinguishes_on_contact) so it hands off to CONSUME. This leg tests whether that
extinguish-and-hand-off binding is what 781 lacked.

FRAMING: RETENTION (BC-install), architect decision 2026-07-25. BC-install the raw_view policy to
its competence band in the CONSUMMATORY env (the consummatory-aware LocalViewGreedyPolicy
demonstrator navigates AND consumes -- consummatory_aware_reference_policies, 2026-07-25), then
run the REFERENCE RL refinement UNDER AN APPROACH DRIVE, and measure whether the installed
competence is RETAINED. The DV is the POST-INSTALLATION COMPETENCE TRAJECTORY (retention probe),
NOT a terminal scalar (terminal-only measurement is what kept this deficit invisible for ten
legs; V3-EXQ-780 is the worked failure). Adjudicated against each arm's OWN post-BC install
(consummatory install band ~13, measured live via the anchors -- NOT the non-consummatory 20.933;
CONSUME costs one step per resource).

ARMS (single manipulation = the drive's extinction; both consummatory, both BC-installed, both
carrying the approach drive):
  * extinct_off (OFF / control) -- consummatory env + non-extinguishing approach drive
    (approach_extinguishes_on_contact=False). This is 781's non-extinguishing terminal drive,
    held in the consummatory env so the ONLY thing that differs from the treatment is extinction.
    In the consummatory env the resource is RETAINED until CONSUME, so a non-extinguishing
    proximity drive rewards camping on the retained cell indefinitely -- a strictly STRONGER
    approach-without-consummation pathology than 781's auto-consume env, i.e. a conservative
    control. Prediction: the installed competence ERODES (the drive fights the installed
    forage/CONSUME behaviour).
  * extinct_on (ON / treatment) -- consummatory env + EXTINGUISHING approach drive
    (approach_extinguishes_on_contact=True). The drive terminates on contact and hands off to
    CONSUME. Prediction: the installed competence is RETAINED.

ANTI-ALIAS (load-bearing). The ONLY thing that differs between the arms is
approach_extinguishes_on_contact. Everything else -- the consummatory env, the approach drive
(use_approach_primitive=True, approach_coef=1.0), the reference build (128-wide, 3x budget,
z_world detached, credit-replay 3/topk 32, bc_aux_coef 0.5), the BC install -- is IDENTICAL
across arms, supplied by baselines.reference_config (+ the approach knobs). use_distributional_
critic, the bc_aux schedule, and the KL-anchor knobs are held at their reference DEFAULTS so this
leg stays orthogonal to the three other manipulation nodes (mech457_distributional_critic = value
estimator / mech457_policy_kl_anchor = update constraint / mech457_bc_aux_schedule = auxiliary
persistence).

DV IS A TRAJECTORY. cfg.retention_probe_every wires the substrate's non-perturbing mid-training
competence probe (train_a2c snapshots/restores the torch/numpy/random streams around every
reading, so measurement neutrality is a substrate guarantee). At 250-episode cadence over the
3000-episode budget that is 12 readings per (arm x seed), each recorded in full.

INTERPRETATION GRID (the "manipulation succeeded and then decayed" branch is mandatory -- the one
V3-EXQ-780 lacked):
  * substrate_not_ready_requeue                 -- an anchor is sub-floor OR the BC install did
      NOT take on a strict majority of seeds in either arm. UNINFORMATIVE about retention;
      NEVER a retention verdict / substrate_ceiling / does_not_support. Requeue.
  * consummation_binding_retains_competence      -- PASS. The extinguishing (consummatory) arm
      holds the installed competence (retained_fraction >= floor on a strict majority of seeds)
      AND beats the non-extinguishing control by the declared margin. 781's drive-side null WAS a
      consummatory-act artefact: an approach drive that extinguishes on contact and hands off to
      CONSUME does not suppress foraging.
  * consummation_binding_succeeded_then_decayed  -- the extinguishing arm's trajectory PEAK held
      at/above the installed competence on a strict majority of seeds but its terminal
      retained_fraction fell below the retention floor. A retention-DYNAMICS finding, not a null.
  * consummation_binding_eroded_under_both       -- THE DECLARED NULL. The installed prior erodes
      identically whether or not the drive extinguishes -> the extinguish-and-hand-off binding is
      NOT the retention mechanism (H-consummation-binding eliminated). Does NOT weaken MECH-457
      (diagnostic). This is the "competence still flat under the extinguish-and-hand-off arm =>
      the binding is not what 781 lacked" reading.
  * consummation_binding_grid_nondiscriminative  -- no arm passed its gate, or the arms/anchors
      do not separate. Not a refutation; re-pose.

ROUTING CONSUMES THE COVARIATES, NOT ONLY THE TERMINAL CRITERION. post_bc_foraging_competence is
load-bearing (selects the requeue branch before any retention branch can be reached), and the
trajectory SHAPE (peak vs terminal) selects between the retained / succeeded-then-decayed /
eroded branches.

BLAST RADIUS. consummatory_act_enabled=True grows the env action_dim 5 -> 6, re-keying every
actor head and BUSTING all cached arm fingerprints for consummatory-ON lineages -- reuse
correctly refuses across the change (expected, not a regression). The edits to
experiments/_lib/** (mech457_approach_extinction + consummatory_aware_reference_policies) also
change substrate_hash, refusing pre-change baseline reuse fleet-wide for future runs.

MINT (mint-as-you-go). Both boot arms emit reuse-ELIGIBLE with a declared config_slice and
include_driver_script_in_hash=False, so a future consummatory-retention consumer can reuse the
minted cells across a different driver. No separate baseline-only mint job (neither sanctioned
exception applies).

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

EXPERIMENT_TYPE = "v3_exq_821_mech457_consummation_binding"
QUEUE_ID = "V3-EXQ-821"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# --- Probe cadence (the measurement constraint the whole leg exists to satisfy) ------------
# 3000-episode reference budget / 250 = 12 readings per (arm x seed). Module constant (not
# derived at call time) because it is fingerprint-declared in the config_slice: a probed and an
# unprobed cell are bit-identical COMPUTATIONS but are not interchangeable ARTIFACTS.
RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# --- Pre-registered retention thresholds (declared; never derived from the run) ------------
RETAINED_FRACTION_FLOOR = 0.5          # >= half the installed competence survives refinement
RETENTION_ARM_MARGIN = 0.15            # extinguishing must beat non-extinguishing in retained frac
MIN_TRAJECTORY_READINGS = 1.5          # need >= 2 readings for a SHAPE (floor, strict >)

# The reference build / install, re-exported from the shared retention lineage module.
REF_REPRESENTATION = baselines.REF_REPRESENTATION          # raw_view ONLY
REF_ACTOR_CRITIC_HIDDEN = baselines.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = baselines.REF_BUDGET_MULTIPLIER      # 3x
POST_BC_INSTALL_FLOOR = baselines.POST_BC_INSTALL_FLOOR      # 1.0 competence floor
PEAK_SUCCESS_TARGET = COMPETENCE_RESOURCE_FLOOR             # peak must clear the install floor

# The one manipulation of THIS leg (781): a constant appetitive-approach coefficient. The drive
# is the same in both arms; only its extinction-on-contact differs.
APPROACH_COEF = 1.0

CFG_KINDS: Tuple[str, ...] = ("extinct_off", "extinct_on")
OFF_KIND = "extinct_off"


def _arm_id(cfg_kind: str) -> str:
    return f"consumbind_{cfg_kind}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(k) for k in CFG_KINDS)
OFF_ARM = _arm_id(OFF_KIND)
ON_ARM = _arm_id("extinct_on")
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


def _env_kwargs() -> Dict[str, Any]:
    """The D3 foraging rung, in CONSUMMATORY mode (contact affords, CONSUME effects). Both boot
    arms AND the readiness anchors run here, so the anchors' denominators match the boot arms'
    env exactly (the consummatory-aware greedy anchors forage; random_walk stays at the floor)."""
    ek = dict(x734._env_kwargs_for_rung(fan.RUNG))
    ek["consummatory_act_enabled"] = True
    return ek


def _make_cfg(extinct: bool, on_budget: int, probe_every: int) -> boot.BootstrapExplorerConfig:
    """Reference RL-refinement config + THIS leg's approach drive, varying ONLY extinction.

    Starts from baselines.reference_config so the reference build (128-wide, 3x budget, z_world
    detached, credit-replay 3/topk 32, bc_aux_coef 0.5, scalar critic, no KL anchor) is a single
    source of truth shared with 788/789 -- then adds the non-extinguishing appetitive approach
    drive to BOTH arms and flips approach_extinguishes_on_contact for the treatment. The dataclass
    is not frozen, so mutating the returned config is safe (reference_config constructs a fresh
    instance per call)."""
    cfg = baselines.reference_config(on_budget, retention_probe_every=int(probe_every))
    cfg.use_approach_primitive = True
    cfg.approach_coef = APPROACH_COEF
    cfg.approach_extinguishes_on_contact = bool(extinct)
    return cfg


def _config_slice(cfg_kind: str, env_kwargs: Dict[str, Any], eval_eps: int, steps: int,
                  on_budget: int, probe_every: int) -> Dict[str, Any]:
    """Declared config_slice for the arm fingerprint. Declares ONLY what the cell computes: the
    consummatory env, eval geometry, the install, the reference build, and this leg's approach +
    extinction knobs (via cfg.as_slice())."""
    cfg = _make_cfg(cfg_kind == "extinct_on", on_budget, probe_every)
    base: Dict[str, Any] = {
        "arm_id": _arm_id(cfg_kind),
        "rung_id": fan.RUNG_ID,
        "kind": "mech457_consummation_binding",
        "env_kwargs": dict(env_kwargs),
        "representation": REF_REPRESENTATION,
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
        "bc_warmstart_episodes": int(baselines.BC_WARMSTART_EPISODES),
        "bc_demonstrator": "local_view_greedy_consummatory",
        "p0_warmup_episodes": 0,
    }
    base.update(cfg.as_slice())
    return base


def _run_boot_cell(cfg_kind: str, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind)
    cfg = _make_cfg(cfg_kind == "extinct_on", on_budget, probe_every)
    rep_agent = baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)

    # Phase 1 -- BC install in the consummatory env (consummatory-aware demonstrator navigates
    # AND consumes), then measure whether it TOOK.
    install = baselines.install_bc_prior(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
    )
    post_bc = float(install["post_bc_foraging_competence"])

    # Phase 2 -- the reference RL refinement UNDER THE APPROACH DRIVE, probed on a cadence. The
    # BC demonstrator persists as the bc_aux auxiliary (reference default 0.5), identical to both
    # arms, so it is not the manipulation.
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
    row["approach_extinguishes_on_contact"] = bool(cfg.approach_extinguishes_on_contact)
    row["use_approach_primitive"] = bool(cfg.use_approach_primitive)
    row["consummatory_act_enabled"] = bool(env_kwargs.get("consummatory_act_enabled", False))
    # FULL per-seed trajectory -- never collapsed to a terminal scalar.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    row["retained_fraction"] = retained
    row["competence_half_life_episodes"] = half_life
    row["peak_cleared_install_floor"] = bool(peak >= PEAK_SUCCESS_TARGET)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["mean_approach_reward_recent"] = float(guard.get("mean_approach_reward_recent", 0.0))
    row["mean_bc_aux_action_match_recent"] = float(guard.get("mean_bc_aux_action_match_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


# ---------------------------------------------------------------------------------------
# Precondition specs (multi-arm gate). Every entry with a numeric measured+threshold so the
# indexer can recompute `met`; all are FLOORS (direction defaults to "lower").
# ---------------------------------------------------------------------------------------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="local_view_greedy_clears_floor_at_d3",
        description=(
            "The consummatory-aware LocalViewGreedyPolicy (reads the SAME 5x5 resource_field_view "
            "and CONSUMEs on contact) forages above the 1.0 floor in the CONSUMMATORY env -- the "
            "positive control that the env is solvable from the local view WITH a consummatory "
            "act. Below-floor means the substrate/env is not ready, NOT that a drive failed."
        ),
        control="consummatory local_view_greedy foraging_competence @D3 (live, ~13.5)",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="greedy_oracle_clears_floor_at_d3",
        description="Consummatory env is floor-achievable with global info (achievability anchor).",
        control="consummatory greedy_oracle foraging_competence @D3 vs the 1.0 floor (live, ~13.0)",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="post_bc_install_took",
        description=(
            "THE LOAD-BEARING READINESS PRECONDITION. The BC install must have TAKEN before RL: "
            "post_bc_foraging_competence (measured pre-RL, post warm-start, in the consummatory "
            "env) clears the 1.0 floor on the WORST seed of this arm. An install that did not "
            "take is UNINFORMATIVE about retention: there is no installed prior to retain, so the "
            "run self-routes substrate_not_ready_requeue rather than any retention verdict."
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
        structural_max=lambda ctx: float(
            int(ctx["n_episodes"]) // max(1, int(ctx["probe_every"]))
        ),
    ),
    PreconditionSpec(
        name="approach_extinction_active",
        description=(
            "The treatment arm must actually be running the extinguish-and-hand-off drive "
            "(approach_extinguishes_on_contact = 1). Scoped OUT of the non-extinguishing CONTROL "
            "arm, where extinction is not merely absent but is the very thing being controlled "
            "against -- asserting it there would make the control's gate structurally un-passable "
            "and collapse the two-arm design."
        ),
        control="cfg.approach_extinguishes_on_contact as constructed for this arm",
        threshold=0.5,
        kind="manipulation_active",
        applies_to=lambda ctx: bool(ctx["approach_extinguishes_on_contact"]),
        applies_note=(
            "treatment arm only -- the non-extinguishing CONTROL is DEFINED by the absence of "
            "extinction, so this precondition is not meaningful for it"
        ),
    ),
)

PRECONDITION_BY_NAME: Dict[str, PreconditionSpec] = {s.name: s for s in PRECONDITION_SPECS}


def _arm_contexts(on_budget: int, probe_every: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": _arm_id(k),
            "cfg_kind": k,
            "approach_extinguishes_on_contact": (k == "extinct_on"),
            "n_episodes": int(on_budget),
            "probe_every": int(probe_every),
        }
        for k in CFG_KINDS
    ]


def run_experiment(seeds: List[int], on_budget: int, eval_eps: int, steps: int,
                   probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-consummation-binding "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep={REF_REPRESENTATION}, consummatory=True, ON_budget={on_budget}, "
        f"probe_every={probe_every}, eval={eval_eps}, steps={steps}; "
        f"approach_coef={APPROACH_COEF}; manipulation=approach_extinguishes_on_contact ONLY)",
        flush=True,
    )
    arm_ctxs = _arm_contexts(on_budget, probe_every)
    # Design-audit BEFORE any compute: refuse a run carrying a structurally unsatisfiable gate.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    env_kwargs = _env_kwargs()
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        is_boot = arm_id in BOOT_ARMS
        if is_boot:
            slice_cfg = _config_slice(cfg_kind, env_kwargs, eval_eps, steps, on_budget,
                                      probe_every)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID,
                         "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor_consummatory"}
        # Mint-as-you-go: both boot arms reuse-ELIGIBLE with a declared slice and no driver in the
        # hash, so a future consummatory-retention consumer can reuse them across a different
        # driver.
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=not is_boot) as cell:
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

    # Anchors first (readiness gate + denominators, in the consummatory env), then boot arms.
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

    # Refuse a run whose install-took anchor its own known-positive control cannot clear. The
    # consummatory install band is ~13 (CONSUME costs a step per resource), so the reachability
    # reference is the LIVE consummatory greedy anchors -- the competence the BC install clones
    # toward -- not the non-consummatory 20.933.
    reachability_cells = [c for c in (local_view_mean, oracle_mean) if c > 0.0] or [0.0]
    anchor_reachability = assert_anchor_reachable(
        anchor_name="post_bc_install_took",
        reference_cells=reachability_cells,
        score_fn=PRECONDITION_BY_NAME["post_bc_install_took"].met_for,
        threshold=float(POST_BC_INSTALL_FLOOR),
        reference_source=(
            "live consummatory anchors this run: local_view_greedy=%0.3f, greedy_oracle=%0.3f "
            "(the consummatory install band the BC demonstrator clones toward)"
            % (local_view_mean, oracle_mean)
        ),
    )
    print(
        f"anchor reachability: {anchor_reachability['anchor_name']} "
        f"reference={anchor_reachability['n_reference_scored_true']}/"
        f"{anchor_reachability['n_reference_cells']} "
        f"reachable={anchor_reachability['reachable']}", flush=True,
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
        n_peak_cleared = int(sum(1 for c in cells if bool(c.get("peak_cleared_install_floor", False))))
        n_retained = int(sum(1 for v in retained_num if v >= RETAINED_FRACTION_FLOOR))
        n_cells = len(cells)
        per_arm_retention[arm_id] = {
            "arm_id": arm_id,
            "approach_extinguishes_on_contact": (cfg_kind == "extinct_on"),
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
            "n_seeds_peak_cleared_install_floor": n_peak_cleared,
            "peak_cleared_strict_majority": bool(n_cells and n_peak_cleared > (n_cells / 2.0)),
            "competence_half_life_episodes_per_seed": [
                c.get("competence_half_life_episodes") for c in cells
            ],
            "mean_approach_reward_recent_per_seed": [
                round(float(c.get("mean_approach_reward_recent", 0.0)), 6) for c in cells
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
        if ctx["approach_extinguishes_on_contact"]:
            measured["approach_extinction_active"] = 1.0
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
    beats_control_by_margin = bool(
        retained_margin is not None and retained_margin >= RETENTION_ARM_MARGIN
    )
    on_succeeded_then_decayed = bool(on_r["peak_cleared_strict_majority"] and not on_retained)
    eroded_under_both = bool(
        (not on_retained) and (not off_retained)
        and (retained_margin is None or abs(retained_margin) < RETENTION_ARM_MARGIN)
        and not on_succeeded_then_decayed
    )
    c_load_bearing = bool(on_retained and beats_control_by_margin)

    if not anchors_ready or not install_took_both:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not gate["non_degenerate"]:
        outcome, label = "FAIL", "consummation_binding_grid_nondiscriminative"
    elif c_load_bearing:
        outcome, label = "PASS", "consummation_binding_retains_competence"
    elif on_succeeded_then_decayed:
        outcome, label = "FAIL", "consummation_binding_succeeded_then_decayed"
    elif eroded_under_both:
        outcome, label = "FAIL", "consummation_binding_eroded_under_both"
    else:
        outcome, label = "FAIL", "consummation_binding_grid_nondiscriminative"

    degeneracy = check_degeneracy({
        "d3_boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        }
    })

    criteria_by_arm = {
        ON_ARM: ["C_extinguishing_drive_retains_installed_competence"],
        OFF_ARM: ["C_non_extinguishing_control_erodes_installed_competence"],
    }
    criteria_nd = arm_criteria_non_degenerate(
        criteria_by_arm, gate,
        extra={
            "C_extinguishing_drive_retains_installed_competence": bool(degeneracy["non_degenerate"]),
            "C_non_extinguishing_control_erodes_installed_competence": bool(degeneracy["non_degenerate"]),
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
            {"name": "C_extinguishing_drive_retains_installed_competence",
             "load_bearing": True,
             "description": (
                 "The extinguishing (consummatory) arm holds a strict majority of seeds at "
                 f"retained_fraction >= {RETAINED_FRACTION_FLOOR} AND beats the non-extinguishing "
                 f"control's mean retained_fraction by >= {RETENTION_ARM_MARGIN}."
             ),
             "passed": c_load_bearing},
            {"name": "C_non_extinguishing_control_erodes_installed_competence",
             "load_bearing": False,
             "description": (
                 "The non-extinguishing control does NOT hold the installed competence -- the "
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
                 "does_not_support. Requeue."
             )},
            {"label": "consummation_binding_retains_competence", "outcome": "PASS",
             "condition": (
                 "the extinguishing arm retains on a strict majority of seeds AND beats the "
                 "non-extinguishing control's mean retained_fraction by the declared margin"
             ),
             "reading": (
                 "781's drive-side null WAS a consummatory-act artefact: an approach drive that "
                 "extinguishes on contact and hands off to CONSUME does not suppress foraging."
             )},
            {"label": "consummation_binding_succeeded_then_decayed", "outcome": "FAIL",
             "condition": (
                 "the extinguishing arm's trajectory PEAK cleared the install floor on a strict "
                 "majority of seeds, but its terminal retained_fraction fell below the retention "
                 "floor"
             ),
             "reading": (
                 "The manipulation SUCCEEDED and then DECAYED -- a retention-DYNAMICS finding, "
                 "not a null. Read the trajectory and the half-life, not the terminal scalar."
             )},
            {"label": "consummation_binding_eroded_under_both", "outcome": "FAIL",
             "condition": (
                 "neither arm retains and their mean retained_fraction differ by less than the "
                 "declared margin"
             ),
             "reading": (
                 "THE DECLARED NULL. The installed prior erodes identically whether or not the "
                 "drive extinguishes -> the extinguish-and-hand-off binding is NOT the retention "
                 "mechanism (H-consummation-binding eliminated). Does NOT weaken MECH-457 "
                 "(diagnostic). This is 'competence still flat under the extinguish-and-hand-off "
                 "arm => the binding is not what 781 lacked'."
             )},
            {"label": "consummation_binding_grid_nondiscriminative", "outcome": "FAIL",
             "condition": (
                 "no arm passed its precondition gate, or the arms/anchors do not separate"
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
            "local_view_greedy_d3_consummatory": round(local_view_mean, 6),
            "greedy_oracle_d3_consummatory": round(oracle_mean, 6),
            "post_bc_worst_by_arm": {
                a: per_arm_retention[a]["post_bc_foraging_competence_worst"] for a in BOOT_ARMS
            },
            "post_bc_worst_seed_by_arm": {
                a: per_arm_retention[a]["post_bc_worst_seed"] for a in BOOT_ARMS
            },
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
            "extinguishing_retains_installed_competence": c_load_bearing,
            "extinguishing_succeeded_then_decayed": on_succeeded_then_decayed,
            "eroded_under_both": eroded_under_both,
            "retained_fraction_mean_extinguishing": on_frac,
            "retained_fraction_mean_non_extinguishing": off_frac,
            "retained_fraction_margin_on_minus_off": retained_margin,
            "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
            "retention_arm_margin": RETENTION_ARM_MARGIN,
            "peak_success_target_install_floor": PEAK_SUCCESS_TARGET,
            "retention_probe_every": int(probe_every),
            "d3_local_view_greedy_consummatory_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle_consummatory": round(oracle_mean, 6),
            "d3_random_walk_consummatory": round(_mean("random_walk"), 6),
        },
        "per_arm_retention": per_arm_retention,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "post_bc_install_floor": float(POST_BC_INSTALL_FLOOR),
            "local_view_greedy_d3_consummatory_live": round(local_view_mean, 6),
            "greedy_oracle_d3_consummatory_live": round(oracle_mean, 6),
            "local_view_greedy_d3_738_reference_non_consummatory": float(fan.DENOM_738_D3_REFERENCE),
            "post_bc_780_reference_raw_view_non_consummatory": 20.933,
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
            "reusable_arms": list(BOOT_ARMS),
            "reuse_eligible": True,
            "note": (
                "Both boot arms are emitted reuse-ELIGIBLE with a declared config_slice and "
                "include_driver_script_in_hash=False, so a future consummatory-retention consumer "
                "can reuse the minted cells across a different driver. No separate baseline-only "
                "mint job (neither sanctioned exception applies)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The post-installation competence TRAJECTORY (not terminal competence) of a "
            "BC-installed raw_view policy across the reference RL refinement in the CONSUMMATORY "
            f"env, probed every {cfg['retention_probe_every']} episodes, under a NON-EXTINGUISHING "
            "vs an EXTINGUISHING appetitive approach drive. Statistic: retained_fraction = "
            "terminal trajectory competence / installed (post-BC) competence, with the trajectory "
            f"PEAK read alongside it. PASS: the extinguishing arm holds retained_fraction >= "
            f"{RETAINED_FRACTION_FLOOR} on a strict majority of seeds AND beats the "
            f"non-extinguishing control's mean by >= {RETENTION_ARM_MARGIN}. Readiness: post_bc_"
            "foraging_competence clears the 1.0 install floor on the worst seed of each arm; the "
            "consummatory install band is ~13 (live anchors), NOT the non-consummatory 20.933."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 RETENTION leg 4 H-consummation-binding, pre-registered under "
            "question 'competence_floor' in hypothesis_space_registry.v1.json. DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy. "
            "FRAMING = retention (BC-install), architect decision 2026-07-25. MANIPULATION = the "
            "drive's EXTINCTION ONLY (approach_extinguishes_on_contact); the consummatory env, the "
            "approach drive (use_approach_primitive, approach_coef=1.0), the reference build "
            "(128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32 / bc_aux 0.5) "
            "and the BC install are IDENTICAL across arms. use_distributional_critic, the bc_aux "
            "schedule and the KL-anchor knobs are HELD AT DEFAULTS across arms (anti-alias with "
            "mech457_distributional_critic / mech457_bc_aux_schedule / mech457_policy_kl_anchor). "
            "SUBSTRATE: mech457_consummatory_act (env, ree-v3 b2e9068) + mech457_approach_extinction "
            "(drive extinction) + consummatory_aware_reference_policies (demonstrator/anchors "
            "CONSUME) all landed 2026-07-25. BLAST RADIUS: consummatory_act_enabled=True grows "
            "action_dim 5->6 and busts cached arm fingerprints for consummatory-ON lineages -- "
            "reuse correctly refuses (expected). DECLARED NULL: the installed prior erodes "
            "identically whether or not the drive extinguishes -> extinction is NOT the retention "
            "mechanism (label consummation_binding_eroded_under_both); does NOT weaken MECH-457, "
            "which stays candidate/v3_pending. GOV-REUSE-1: the decisive readout (retained_fraction "
            "of an approach-driven consummatory-env refinement) is recorded in NO existing "
            "manifest (the consummatory action space is new as of 2026-07-25), so this must run."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-821 MECH-457 GOV-FANOUT-1 H-consummation-binding"
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
        "consummatory_act_enabled": True,
        "on_budget_episodes": on_budget,
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "retention_probe_every": probe_every,
        "approach_coef": APPROACH_COEF,
        "approach_signal": "resource_field_view_peak",
        "bc_warmstart_episodes": int(baselines.BC_WARMSTART_EPISODES),
        "bc_aux_coef": float(baselines.BC_AUX_COEF_BASELINE),
        "bc_demonstrator": "local_view_greedy_consummatory",
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": baselines.REF_CREDIT_PASSES,
        "ref_credit_topk": baselines.REF_CREDIT_TOPK,
        "ref_cotrain_encoder": baselines.REF_COTRAIN_ENCODER,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA, "bc_lr": fan.BC_LR,
        "extinct_off_config": _make_cfg(False, on_budget, probe_every).as_slice(),
        "extinct_on_config": _make_cfg(True, on_budget, probe_every).as_slice(),
        "retained_fraction_floor": RETAINED_FRACTION_FLOOR,
        "retention_arm_margin": RETENTION_ARM_MARGIN,
        "peak_success_target_install_floor": PEAK_SUCCESS_TARGET,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 retention (H-consummation-binding, leg 4)",
        "hypothesis_id": "H-consummation-binding",
        "hypothesis_question": "competence_floor",
        "framing": "retention_bc_install",
        "substrate_nodes": [
            "mech457_consummatory_act", "mech457_approach_extinction",
            "consummatory_aware_reference_policies",
        ],
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
    hl = result["headline"]
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
