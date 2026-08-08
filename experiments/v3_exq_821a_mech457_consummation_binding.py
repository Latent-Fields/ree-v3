#!/opt/local/bin/python3
"""V3-EXQ-821a -- MECH-457 GOV-FANOUT-1 RETENTION leg 4, H-consummation-binding: CALIBRATED-INSTALL
re-run of V3-EXQ-821 (supersedes it).

WHY THIS SUPERSEDES 821. V3-EXQ-821 (2026-07-25) came back FAIL / consummation_binding_eroded_
under_both, but competence_floor_instrument_audit_2026-08-07.md section 3d showed that reading
was an OBSERVATION-BOTTLENECK artefact, not a discrimination: both arms installed at only ~14% of
the run's own consummatory anchor ceiling (post_bc 4.3-5.7 against a live local_view_greedy of
34.17; greedy_oracle 42.8), then collapsed to foraging_competence EXACTLY 0.000, goal_reach_rate
0.000, survival 17.8/15.0 (BELOW random_walk's 34.0) on every sub-skill of every seed. Two arms
tied at the absolute floor cannot discriminate anything, so the leg's resolution.state was reset
from `eliminated` back to `alive` (governance-applied 2026-08-07), and competence_floor_recurrence_
repose_2026-08-08.md re-posed it as a CALIBRATED single re-run (this experiment).

ROOT CAUSE OF 821's WEAK INSTALL (diagnosed here, 2026-08-08). The BC warm-start is unweighted
cross-entropy over the consummatory-aware demonstrator's actions. In the consummatory env CONSUME
is a distinct 6th action emitted only when standing ON a resource -- a RARE but CRITICAL action.
At 821's default n_bc=300 the clone reached ~0.86 overall action-match (dominated by the abundant
navigation actions) but UNDER-cloned the rare CONSUME, so it navigated onto resources and failed
to consume them -> forage collapsed to ~14% of the demonstrator despite 86% action-match. This is
a class-imbalance / rare-critical-action install defect, NOT a wrong-demonstrator bug (the clone
DOES consume sometimes -> the demonstrator is consummatory-aware; confirmed cc15399 2026-07-25)
and NOT a new retention mechanism (so competence_floor's growth_restriction stop-condition is not
triggered; this is the sanctioned calibrated re-run of the one already-alive leg).

THE FIX (empirically calibrated 2026-08-08, seed 42, consummatory env, this exact install path):
    dose  post_bc  action_match
     300    3.10     0.857     <- 821's regime (degenerate)
    1200   25.05     0.961     <- lands in the campaign's usable 20.9-38.3 forage/episode band
    2400   21.05     0.960     <- saturated; no further gain
The install SATURATES by n_bc=1200, so INSTALL_DOSE=1200 is the calibrated point: it clears a
usable floor with margin and going higher does not help. Threaded via install_bc_prior(n_bc=...),
the dose knob added 2026-07-22 for the V3-EXQ-836 dose-sweep.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PRE-REGISTERED as hypothesis H-consummation-binding under question qid "competence_floor" in
REE_assembly/evidence/planning/hypothesis_space_registry.v1.json (mech457_retention_portfolio
2026-07-18, leg 4; competence_floor_reposing 2026-07-25; instrument audit 2026-08-07;
recurrence re-pose 2026-08-08). This is a NEW adjudicating run on the EXISTING leg -- registered
via adjudicating_runs (Mode A, existing-question path; NO denominator change).

THE HYPOTHESIS (unchanged from 821). V3-EXQ-781 gave the composed bootstrap an innate,
NON-EXTINGUISHING appetitive approach drive and found the drive EARNED (0.707) while raw-view
foraging was SUPPRESSED to 0.200 from a 2.983 control -- approach WITHOUT consummation. The
hypothesis is that 781's drive-side null was itself an artefact of a MISSING CONSUMMATORY ACT: its
terminal drive was non-extinguishing, so there was no mechanism for the approach drive to
terminate correctly on arrival and hand off to a distinct act of consuming. The
mech457_consummatory_act env node (2026-07-25) makes contact AFFORD rather than EFFECT consumption
(a distinct CONSUME action), and mech457_approach_extinction (2026-07-25) lets the approach drive
EXTINGUISH on contact (approach_extinguishes_on_contact) so it hands off to CONSUME. This leg tests
whether that extinguish-and-hand-off binding is what 781 lacked.

FRAMING: RETENTION (BC-install), architect decision 2026-07-25. BC-install the raw_view policy to
its competence band in the CONSUMMATORY env (the consummatory-aware LocalViewGreedyPolicy
demonstrator navigates AND consumes -- consummatory_aware_reference_policies, 2026-07-25) AT THE
CALIBRATED DOSE, then run the REFERENCE RL refinement UNDER AN APPROACH DRIVE, and measure whether
the installed competence is RETAINED. The DV is the POST-INSTALLATION COMPETENCE TRAJECTORY
(retention probe), NOT a terminal scalar.

INSTRUMENT FIX (competence_floor_instrument_audit_2026-08-07 sections 2/7; landed ree-v3 0550a2f).
make_probe_fn and install_bc_prior now return the FULL 8-metric evaluate_seed row (not just the
foraging_competence projection), so this run reports foraging RATE (resources/TICK) alongside the
composite resources/EPISODE count -- the composite conflates foraging rate with episode duration,
which are anti-correlated at the floor in this env (the most survivable policy is the least
competent). survival_horizon is reported as a COVARIATE, not a sub-skill. rate/tick =
foraging_competence / survival_horizon.

USABLE INSTALL FLOOR (the second half of the audit's fix). 821's readiness gate accepted any
install clearing the trivial 1.0 competence floor, so 4.3 counted as "install took" and the run
produced a degenerate two-arms-at-zero comparison. Here the readiness gate requires the install to
clear USABLE_INSTALL_FLOOR (10.0) -- comfortably above the degenerate ~3-5 regime and below the
calibrated ~21-25 install, with per-seed margin. An install below it self-routes
substrate_not_ready_requeue (informative: the calibrated dose did not take on some seed ->
install-mechanism work, an /implement-substrate finding) rather than a spurious retention null.

ARMS (single manipulation = the drive's extinction; both consummatory, both BC-installed at the
calibrated dose, both carrying the approach drive):
  * extinct_off (OFF / control) -- consummatory env + non-extinguishing approach drive
    (approach_extinguishes_on_contact=False). This is 781's non-extinguishing terminal drive,
    held in the consummatory env so the ONLY thing that differs from the treatment is extinction.
    In the consummatory env the resource is RETAINED until CONSUME, so a non-extinguishing
    proximity drive rewards camping on the retained cell indefinitely -- a strictly STRONGER
    approach-without-consummation pathology than 781's auto-consume env, i.e. a conservative
    control. Prediction: the installed competence ERODES.
  * extinct_on (ON / treatment) -- consummatory env + EXTINGUISHING approach drive
    (approach_extinguishes_on_contact=True). The drive terminates on contact and hands off to
    CONSUME. Prediction: the installed competence is RETAINED.

ANTI-ALIAS (load-bearing). The ONLY thing that differs between the arms is
approach_extinguishes_on_contact. Everything else -- the consummatory env, the approach drive
(use_approach_primitive=True, approach_coef=1.0), the reference build (128-wide, 3x budget,
z_world detached, credit-replay 3/topk 32, bc_aux_coef 0.5), the BC install dose -- is IDENTICAL
across arms, supplied by baselines.reference_config (+ the approach knobs). use_distributional_
critic, the bc_aux schedule, and the KL-anchor knobs are held at their reference DEFAULTS so this
leg stays orthogonal to the three other manipulation nodes (mech457_distributional_critic = value
estimator / mech457_policy_kl_anchor = update constraint / mech457_bc_aux_schedule = auxiliary
persistence).

DV IS A TRAJECTORY. cfg.retention_probe_every wires the substrate's non-perturbing mid-training
competence probe (train_a2c snapshots/restores the torch/numpy/random streams around every
reading, so measurement neutrality is a substrate guarantee). At 250-episode cadence over the
3000-episode budget that is 12 readings per (arm x seed), each recorded in full (all 8 metrics).

INTERPRETATION GRID (the "manipulation succeeded and then decayed" branch is mandatory):
  * substrate_not_ready_requeue                 -- an anchor is sub-floor OR the calibrated BC
      install did NOT clear USABLE_INSTALL_FLOOR (10.0) on a strict majority of seeds in either
      arm. UNINFORMATIVE about retention; NEVER a retention verdict / substrate_ceiling /
      does_not_support. Requeue (dose/install-mechanism work).
  * consummation_binding_retains_competence      -- PASS. The extinguishing (consummatory) arm
      holds the installed competence (retained_fraction >= floor on a strict majority of seeds)
      AND beats the non-extinguishing control by the declared margin. 781's drive-side null WAS a
      consummatory-act artefact.
  * consummation_binding_succeeded_then_decayed  -- the extinguishing arm's trajectory PEAK held
      at/above the install floor on a strict majority of seeds but its terminal retained_fraction
      fell below the retention floor. A retention-DYNAMICS finding, not a null.
  * consummation_binding_eroded_under_both       -- THE DECLARED NULL. The installed prior erodes
      identically whether or not the drive extinguishes -> the extinguish-and-hand-off binding is
      NOT the retention mechanism (H-consummation-binding eliminated). Does NOT weaken MECH-457
      (diagnostic).
  * consummation_binding_grid_nondiscriminative  -- no arm passed its gate, or the arms/anchors
      do not separate. Not a refutation; re-pose.

ROUTING CONSUMES THE COVARIATES, NOT ONLY THE TERMINAL CRITERION. post_bc_foraging_competence (vs
the USABLE floor) is load-bearing (selects the requeue branch before any retention branch can be
reached), and the trajectory SHAPE (peak vs terminal) selects between the retained /
succeeded-then-decayed / eroded branches.

BLAST RADIUS. consummatory_act_enabled=True grows the env action_dim 5 -> 6, re-keying every actor
head and BUSTING all cached arm fingerprints for consummatory-ON lineages -- reuse correctly
refuses across the change (expected). The calibrated install dose (1200 vs 821's 300) also changes
the config_slice (bc_warmstart_episodes), so 821a's boot cells are a DISTINCT fingerprint from
821's -- they are not interchangeable ARTIFACTS (different install), which is correct.

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

EXPERIMENT_TYPE = "v3_exq_821a_mech457_consummation_binding"
QUEUE_ID = "V3-EXQ-821a"
SUPERSEDES = "V3-EXQ-821"
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

# --- Calibrated BC install dose (THE 821a FIX) ---------------------------------------------
# 821 used the lib default (300), which under-clones the rare CONSUME action in the consummatory
# env and lands the install at ~14% of ceiling (post_bc 4.3-5.7). Empirical calibration
# 2026-08-08 (seed 42, this exact install path): dose 300 -> 3.10, 1200 -> 25.05, 2400 -> 21.05
# (saturated). 1200 clears the campaign's usable 20.9-38.3 band with margin.
INSTALL_DOSE = 1200
DRY_INSTALL_DOSE = 4                    # tiny dose so --dry-run smoke is fast

# --- Pre-registered retention thresholds (declared; never derived from the run) ------------
RETAINED_FRACTION_FLOOR = 0.5          # >= half the installed competence survives refinement
RETENTION_ARM_MARGIN = 0.15            # extinguishing must beat non-extinguishing in retained frac
MIN_TRAJECTORY_READINGS = 1.5          # need >= 2 readings for a SHAPE (floor, strict >)

# USABLE install floor (the 821a readiness fix). An install must clear THIS, not the trivial 1.0
# competence floor, to be a meaningful retention substrate. Sits above the degenerate ~3-5 regime
# 821 accepted and below the calibrated ~21-25 install, with per-seed margin.
USABLE_INSTALL_FLOOR = 10.0

# The reference build / install, re-exported from the shared retention lineage module.
REF_REPRESENTATION = baselines.REF_REPRESENTATION          # raw_view ONLY
REF_ACTOR_CRITIC_HIDDEN = baselines.REF_ACTOR_CRITIC_HIDDEN  # 128
REF_BUDGET_MULTIPLIER = baselines.REF_BUDGET_MULTIPLIER      # 3x
# NOTE: script-local POST_BC_INSTALL_FLOOR is the USABLE floor (10.0), NOT the lib's 1.0. The
# readiness gate, routing, and anchor-reachability all use this; the lib's internal install_took
# (1.0) is overridden per cell below.
POST_BC_INSTALL_FLOOR = USABLE_INSTALL_FLOOR
PEAK_SUCCESS_TARGET = COMPETENCE_RESOURCE_FLOOR             # peak must clear the 1.0 install floor

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


def _rate_per_tick(row: Dict[str, Any]) -> float:
    """Foraging RATE = resources / TICK = foraging_competence / survival_horizon (the audit's
    de-confounded DV). survival_horizon is a COVARIATE, not a sub-skill; 0 -> rate 0."""
    surv = float(row.get("survival_horizon", 0.0) or 0.0)
    if surv <= 0.0:
        return 0.0
    return round(float(row.get("foraging_competence", 0.0)) / surv, 6)


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
                  on_budget: int, probe_every: int, install_dose: int) -> Dict[str, Any]:
    """Declared config_slice for the arm fingerprint. Declares ONLY what the cell computes: the
    consummatory env, eval geometry, the install (dose is load-bearing -- a 1200-dose install is a
    DISTINCT artifact from a 300-dose one), the reference build, and this leg's approach +
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
        "bc_warmstart_episodes": int(install_dose),   # CALIBRATED dose, not the lib default
        "bc_demonstrator": "local_view_greedy_consummatory",
        "p0_warmup_episodes": 0,
    }
    base.update(cfg.as_slice())
    return base


def _run_boot_cell(cfg_kind: str, env_kwargs: Dict[str, Any], seed: int, on_budget: int,
                   eval_eps: int, steps: int, probe_every: int,
                   install_dose: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind)
    cfg = _make_cfg(cfg_kind == "extinct_on", on_budget, probe_every)
    rep_agent = baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)

    # Phase 1 -- BC install in the consummatory env at the CALIBRATED DOSE (consummatory-aware
    # demonstrator navigates AND consumes), then measure whether it TOOK against the USABLE floor.
    install = baselines.install_bc_prior(
        rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id,
        n_bc=int(install_dose),
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
    row["bc_install_dose"] = int(install.get("bc_install_dose", install_dose))
    # 821a: install_took is judged against the USABLE floor, NOT the lib's trivial 1.0 floor. An
    # install that cleared 1.0 but not USABLE_INSTALL_FLOOR is uninformative about retention.
    row["install_took"] = bool(post_bc >= POST_BC_INSTALL_FLOOR)
    row["install_took_usable_floor"] = float(POST_BC_INSTALL_FLOOR)
    row["install_took_lib_1p0_floor"] = bool(install["install_took"])   # provenance only
    row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
    # FULL post-BC evaluate_seed row (all 8 capability metrics) -- the audit's discarded-sub-skills
    # fix (0550a2f). Plumbed here (821 predated it) so post-BC rate/tick + sub-skills are recorded.
    row["post_bc_capability_row"] = dict(install.get("post_bc_capability_row", {}))
    row["post_bc_rate_per_tick"] = _rate_per_tick(install.get("post_bc_capability_row", {}))
    row["approach_extinguishes_on_contact"] = bool(cfg.approach_extinguishes_on_contact)
    row["use_approach_primitive"] = bool(cfg.use_approach_primitive)
    row["consummatory_act_enabled"] = bool(env_kwargs.get("consummatory_act_enabled", False))
    # FULL per-seed trajectory -- never collapsed to a terminal scalar. Each row now carries all 8
    # metrics (0550a2f), so the trajectory is sub-skill decomposable.
    row["competence_trajectory"] = trajectory
    row["n_trajectory_readings"] = int(len(trajectory))
    row["trajectory_peak_competence"] = peak
    row["trajectory_terminal_competence"] = terminal_traj
    # De-confounded rate DV (audit sections 2/7): resources/TICK on the terminal eval + trajectory.
    row["rate_per_tick"] = _rate_per_tick(row)
    traj_rate_vals = [_rate_per_tick(r) for r in trajectory]
    row["trajectory_rate_per_tick_peak"] = round(max(traj_rate_vals), 6) if traj_rate_vals else 0.0
    row["trajectory_rate_per_tick_terminal"] = (
        round(traj_rate_vals[-1], 6) if traj_rate_vals else 0.0
    )
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
        control="consummatory local_view_greedy foraging_competence @D3 (live, ~34)",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="greedy_oracle_clears_floor_at_d3",
        description="Consummatory env is floor-achievable with global info (achievability anchor).",
        control="consummatory greedy_oracle foraging_competence @D3 vs the 1.0 floor (live, ~42)",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        kind="readiness",
    ),
    PreconditionSpec(
        name="post_bc_install_took",
        description=(
            "THE LOAD-BEARING READINESS PRECONDITION. The CALIBRATED BC install must have taken to "
            "a USABLE level before RL: post_bc_foraging_competence (measured pre-RL, post "
            "warm-start, in the consummatory env) clears USABLE_INSTALL_FLOOR (10.0) on the WORST "
            "seed of this arm. 821's bug was accepting a 4.3 install (cleared 1.0 but useless), "
            "producing a degenerate two-arms-at-zero comparison. An install below the usable floor "
            "is UNINFORMATIVE about retention -- there is no usable installed prior to retain -- so "
            "the run self-routes substrate_not_ready_requeue rather than any retention verdict."
        ),
        control="post-BC / pre-RL foraging_competence of this arm's worst seed vs the 10.0 usable floor",
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
                   probe_every: int, install_dose: int,
                   dry_run: bool = False) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-consummation-binding (821a CALIBRATED re-run) "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"rep={REF_REPRESENTATION}, consummatory=True, ON_budget={on_budget}, "
        f"install_dose={install_dose}, usable_install_floor={USABLE_INSTALL_FLOOR}, "
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
                                      probe_every, install_dose)
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
                                     probe_every, install_dose)
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

    # Refuse a run whose USABLE install-took floor its own known-positive control cannot clear. The
    # consummatory demonstrator hits ~34-42 (live anchors on a real run), so the reachability
    # reference is those live anchors -- the competence the BC install clones toward. SKIPPED under
    # --dry-run: the toy dry env (steps=15, eval=2) caps the anchors at ~2.5, which cannot clear
    # the real-run usable floor (10.0) -- a scale artifact, not a design fault, so asserting it
    # there would wrongly refuse the smoke. The guard is meaningful only at real scale.
    if dry_run:
        anchor_reachability = {
            "anchor_name": "post_bc_install_took", "reachable": None,
            "n_reference_scored_true": None, "n_reference_cells": len([local_view_mean, oracle_mean]),
            "skipped_reason": "dry_run: toy-scale anchors cannot clear the real-run usable floor",
        }
        print("anchor reachability: SKIPPED under --dry-run (toy-scale anchors)", flush=True)
    else:
        reachability_cells = [c for c in (local_view_mean, oracle_mean) if c > 0.0] or [0.0]
        anchor_reachability = assert_anchor_reachable(
            anchor_name="post_bc_install_took",
            reference_cells=reachability_cells,
            score_fn=PRECONDITION_BY_NAME["post_bc_install_took"].met_for,
            threshold=float(POST_BC_INSTALL_FLOOR),
            reference_source=(
                "live consummatory anchors this run: local_view_greedy=%0.3f, greedy_oracle=%0.3f "
                "(the consummatory install band the BC demonstrator clones toward; usable floor %0.1f)"
                % (local_view_mean, oracle_mean, POST_BC_INSTALL_FLOOR)
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
            "post_bc_rate_per_tick_per_seed": [
                float(c.get("post_bc_rate_per_tick", 0.0)) for c in cells
            ],
            "bc_install_dose": int(install_dose),
            "usable_install_floor": float(POST_BC_INSTALL_FLOOR),
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
            # De-confounded rate DV (audit sections 2/7).
            "terminal_rate_per_tick_per_seed": [
                float(c.get("rate_per_tick", 0.0)) for c in cells
            ],
            "terminal_rate_per_tick_mean": round(
                float(sum(float(c.get("rate_per_tick", 0.0)) for c in cells) / n_cells), 6
            ) if n_cells else 0.0,
            "survival_horizon_per_seed": [
                float(c.get("survival_horizon", 0.0)) for c in cells
            ],
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
                 "an anchor is sub-floor, OR post_bc_foraging_competence fails the "
                 f"{POST_BC_INSTALL_FLOOR} USABLE install floor on a strict majority of seeds in "
                 "either arm"
             ),
             "reading": (
                 "There is no usable installed prior to retain, so the run is UNINFORMATIVE about "
                 "retention (this is exactly the 821 degenerate-install failure this re-run fixes). "
                 "NEVER a retention verdict and never substrate_ceiling / does_not_support. "
                 "Requeue (dose / install-mechanism work)."
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
                 "(diagnostic). Unlike 821, this reading is now made against a USABLE install "
                 "(post_bc >= 10) so a null is a real erosion, not an install artefact."
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
            "usable_install_floor": float(POST_BC_INSTALL_FLOOR),
            "bc_install_dose": int(install_dose),
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
                    "The CALIBRATED BC install must have TAKEN to a USABLE level before RL on "
                    "every arm: post-BC / pre-RL foraging_competence of the worst cell clears the "
                    "10.0 usable install floor. Below it there is no usable installed prior to "
                    "retain and the run self-routes substrate_not_ready_requeue."
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
            "usable_install_floor": float(POST_BC_INSTALL_FLOOR),
            "bc_install_dose": int(install_dose),
            "retention_probe_every": int(probe_every),
            # De-confounded rate DV (audit sections 2/7) reported alongside the composite.
            "terminal_rate_per_tick_mean_extinguishing": on_r["terminal_rate_per_tick_mean"],
            "terminal_rate_per_tick_mean_non_extinguishing": off_r["terminal_rate_per_tick_mean"],
            "d3_local_view_greedy_consummatory_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle_consummatory": round(oracle_mean, 6),
            "d3_random_walk_consummatory": round(_mean("random_walk"), 6),
        },
        "per_arm_retention": per_arm_retention,
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "usable_install_floor": float(POST_BC_INSTALL_FLOOR),
            "local_view_greedy_d3_consummatory_live": round(local_view_mean, 6),
            "greedy_oracle_d3_consummatory_live": round(oracle_mean, 6),
            "local_view_greedy_d3_738_reference_non_consummatory": float(fan.DENOM_738_D3_REFERENCE),
            "post_bc_780_reference_raw_view_non_consummatory": 20.933,
            "calibration_2026_08_08": {
                "seed": 42, "dose_300_post_bc": 3.1, "dose_1200_post_bc": 25.05,
                "dose_2400_post_bc": 21.05, "chosen_dose": int(install_dose),
            },
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
                "can reuse the minted cells across a different driver. Distinct fingerprint from "
                "821 (calibrated install dose 1200 vs 300). No separate baseline-only mint job."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "The post-installation competence TRAJECTORY (not terminal competence) of a "
            "CALIBRATED-DOSE (n_bc=1200) BC-installed raw_view policy across the reference RL "
            "refinement in the CONSUMMATORY env, probed every "
            f"{cfg['retention_probe_every']} episodes, under a NON-EXTINGUISHING vs an "
            "EXTINGUISHING appetitive approach drive. Statistic: retained_fraction = terminal "
            "trajectory competence / installed (post-BC) competence, with the trajectory PEAK and "
            "the de-confounded foraging RATE (resources/tick) read alongside it. PASS: the "
            f"extinguishing arm holds retained_fraction >= {RETAINED_FRACTION_FLOOR} on a strict "
            f"majority of seeds AND beats the non-extinguishing control's mean by >= "
            f"{RETENTION_ARM_MARGIN}. Readiness: post_bc_foraging_competence clears the "
            f"{POST_BC_INSTALL_FLOOR} USABLE install floor on the worst seed of each arm (821 "
            "accepted 4.3 against a trivial 1.0 floor -> degenerate; calibration lands ~21-25)."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 RETENTION leg 4 H-consummation-binding, CALIBRATED re-run "
            "superseding V3-EXQ-821. Pre-registered under question 'competence_floor' in "
            "hypothesis_space_registry.v1.json (adjudicating_runs, Mode A existing-leg). "
            "DIAGNOSTIC (excluded from scoring); PROMOTES/DEMOTES NOTHING; route to "
            "/failure-autopsy. 821 FIX: (1) BC install dose 300->1200 (calibrated 2026-08-08: "
            "300->post_bc 3.1, 1200->25.05, 2400->21.05 saturated; the rare CONSUME action is "
            "under-cloned at low dose in the 6-action consummatory env, an install class-imbalance "
            "defect NOT a new mechanism); (2) USABLE install readiness floor 10.0 not the trivial "
            "1.0 (821 accepted a 4.3 install and produced a degenerate two-arms-at-zero "
            "comparison); (3) foraging RATE (resources/tick) reported alongside the composite "
            "resources/episode, survival_horizon as a covariate (instrument audit 2026-08-07 "
            "sections 2/7; probe/install full-8-metric fix ree-v3 0550a2f). MANIPULATION = the "
            "drive's EXTINCTION ONLY (approach_extinguishes_on_contact); consummatory env, approach "
            "drive, reference build and BC install dose IDENTICAL across arms. SUBSTRATE: "
            "mech457_consummatory_act + mech457_approach_extinction + "
            "consummatory_aware_reference_policies (all landed 2026-07-25, BUILT+VALIDATED). "
            "RE-DERIVE BRAKE: released -- diagnostic re-test of a currently-ALIVE leg (821's "
            "eliminated reading voided as an instrument artefact, audit sec 3d), upstream substrate "
            "built+validated, instrument fixed. growth_restriction: this is the sanctioned "
            "calibrated re-run of the one alive leg, not a new fan-out. GOV-REUSE-1: the decisive "
            "readout (retained_fraction of a USABLE-install consummatory-env approach-driven "
            "refinement with rate/tick) is recorded in NO manifest -- 821's install was degenerate "
            "and the consummatory action space is new as of 2026-07-25 -- so this must run."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-821a MECH-457 GOV-FANOUT-1 H-consummation-binding (calibrated re-run)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None,
                        help="RL refinement budget (episodes) per arm x seed")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--install-dose", type=int, default=None,
                        help="BC install dose (n_bc); default calibrated 1200 (dry 4)")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        on_budget = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
        probe_every = DRY_RETENTION_PROBE_EVERY
        install_dose = DRY_INSTALL_DOSE
    else:
        seeds = list(fan.SEEDS)
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)  # 3000 -- reference budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE
        probe_every = RETENTION_PROBE_EVERY
        install_dose = INSTALL_DOSE

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
    if args.install_dose is not None:
        install_dose = int(args.install_dose)

    result = run_experiment(seeds=seeds, on_budget=on_budget, eval_eps=eval_eps, steps=steps,
                            probe_every=probe_every, install_dose=install_dose,
                            dry_run=bool(args.dry_run))

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
        "bc_warmstart_episodes": int(install_dose),      # CALIBRATED dose (821a fix)
        "bc_install_dose": int(install_dose),
        "usable_install_floor": float(USABLE_INSTALL_FLOOR),
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
        "portfolio": "GOV-FANOUT-1 MECH-457 retention (H-consummation-binding, leg 4, calibrated)",
        "hypothesis_id": "H-consummation-binding",
        "hypothesis_question": "competence_floor",
        "framing": "retention_bc_install",
        "supersedes": SUPERSEDES,
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
            f"terminal={r['terminal_forage_mean']} "
            f"rate/tick={r['terminal_rate_per_tick_mean']}", flush=True,
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
