#!/opt/local/bin/python3
"""V3-EXQ-836 -- MECH-476 falsifier (DOSE axis): is competence RETENTION a dose-dependent
CONSOLIDATION PROCESS, or only a fixed erosion proportion?

CLAIM UNDER TEST. MECH-476 (candidate / v3_pending, split_from MECH-457): acquiring competence
and retaining it are dissociable capabilities, and -- its distinctive, testable content --
consolidation is properly defined as RESISTANCE TO RETROGRADE INTERFERENCE (Krakauer, Ghez &
Ghilardi 2005, J Neurosci 25:473), hence a process that should strengthen with install DOSE and
elapsed offline INTERVAL. V3-EXQ-780 ran a SINGLE BC dose straight into RL with no interval, so
REE has never tested for a consolidation process at all -- only for concurrent regularisation
(788 critic, 792 KL anchor).

WHAT THIS LEG TESTS, AND WHAT IT DELIBERATELY DOES NOT. This is the BUILDABLE axis of the
three-arm falsifier: INSTALL DOSE. It varies the number of behavioural-cloning install episodes
and holds the interfering unconstrained-RL refinement FIXED, then measures the retained-fraction
TRAJECTORY. The other two arms of MECH-476's falsifier -- an A->B offline INTERVAL as a
consolidation PROCESS, and a behavioural-tagging NOVELTY pairing (Moncada & Viola 2007) inside
that window -- require an offline policy-consolidation window the substrate does NOT have
(SLEEP DRIVER: none; the only offline compute is credit-replay inside RL). Those are routed to
/implement-substrate (natural home: sleep_substrate_plan.md SD-017 / MECH-204, per MECH-476's own
cross-link), NOT queued here. This leg alone is a complete test of the claim's LOAD-BEARING
dose-dependence prediction.

HYPOTHESIS (Krakauer's over-training result, ported). Krakauer 2005 experiment 3: DOUBLING the
amount of initial training produced resistance to a 5-min counter-rotation where the standard
dose did not. Ported: a BC-installed policy trained with MORE install episodes should retain a
LARGER fraction of its installed competence through the same unconstrained RL refinement. All
doses sit AT or ABOVE the ~20.9 install point (780) so the install-took confound is avoided and
the question is purely whether over-training deepens the trace.

DESIGN. Three arms, all BC-install the raw_view policy through the SHARED lineage baseline
(baselines.mech457_retention: build_off_arm -> install_bc_prior -> train_off_arm), then run the
IDENTICAL unconstrained reference RL refinement (128-wide / 3x budget / z_world detached /
credit-replay 3 / topk 32, NO KL anchor, NO distributional critic). The ONLY thing that varies is
the install dose:
  * dose_bc300  -- 300 BC episodes (the ~20.9 install point 780 measured; the lineage OFF dose).
  * dose_bc600  -- 600 BC episodes (Krakauer's DOUBLED-training dose).
  * dose_bc900  -- 900 BC episodes (3x; the widest-spaced point for a readable monotone trend).

DV -- a TRAJECTORY, never a terminal scalar. retention_probe_every wires the substrate's
mid-training competence probe; the DV is retained_fraction = terminal_trajectory_competence /
post_bc_installed_competence (baselines.retained_fraction; None when the install was ~0 so an
un-taken install cannot manufacture a maximal-erosion reading).

SCORING (pre-registered).
  * SUPPORTED (outcome PASS, evidence supports MECH-476): over the arms whose install TOOK,
    retained_fraction_mean is monotone non-decreasing in dose AND
    (retained_fraction_mean[max_dose] - retained_fraction_mean[min_dose]) >= RESISTANCE_DOSE_MARGIN.
    Retention is then a dose-dependent consolidation PROCESS.
  * WEAKENED (outcome FAIL, evidence weakens MECH-476 -> withdraw into MECH-459/460): the retained
    fraction is INVARIANT to dose (spread < RESISTANCE_DOSE_MARGIN). No consolidation process;
    retention tracks only the concurrent constraint, exactly the intended failure mode MECH-476
    declares up front.
  * NON-MONOTONE (outcome FAIL, evidence mixed): spread >= margin but not monotone -- a real but
    uninterpretable dose effect; routes to /failure-autopsy rather than a clean verdict.

PREREQUISITE (NON-NEGOTIABLE, inherited from the retention portfolio). An install that did not
take is UNINFORMATIVE about retention. Each arm carries a per-arm readiness precondition
(install_took strict majority of seeds); an arm below it REDS ONLY ITSELF and cannot vacate a
sibling (the V3-EXQ-785 per-arm discipline via precondition_gate). If FEWER THAN TWO dose arms
achieve majority install_took the dose-response is UNMEASURABLE and the whole run self-routes
substrate_not_ready_requeue -- NOT a WEAKENED verdict. Measured invariance across arms that DID
take is a real WEAKENED finding and is NOT degeneracy; the two are kept distinct.

DV-SYMMETRY (mandatory declaration, per arm). The DV is retained_fraction, a competence ratio.
The manipulation is BC install dose (number of BC episodes). retained_fraction is NOT invariant
under "add BC episodes": more install training changes the installed policy's parameters and --
per the hypothesis under test -- its robustness to subsequent erosion. So the manipulation is not
a symmetry of the DV, for every arm. This is a genuine measurement, not an arithmetic identity.

EXPERIMENT_PURPOSE = evidence. This directly tests MECH-476's load-bearing prediction and can move
the claim toward supported/weakened. Single claim (MECH-476), so no evidence_direction_per_claim.

ethics_preflight: all-false / allow. No negative valence, no suffering-like state, no self-model,
no offline replay over harm, no social mind / language, no human/clinical data. V3
pre-ethical instrumentation only (SENT-0).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.baselines.mech457_retention as baselines  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_836_mech476_dose_dependent_consolidation"
QUEUE_ID = "V3-EXQ-836"
CLAIM_IDS: List[str] = ["MECH-476"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Install-dose sweep (the ONLY manipulation) --------------------------------------------
# 300 = the ~20.9 install point 780 measured; 600 = Krakauer's doubled-training dose;
# 900 = 3x, the widest-spaced third point. All >= the took-point so install_took is not the axis.
DOSES: Tuple[int, ...] = (300, 600, 900)
DRY_DOSES: Tuple[int, ...] = (2, 4, 6)

# 792a raised n to 6 after the n=3 lesson (a single anomalous cell IS the arm). Same here: the
# dose-response is a monotone trend that needs seeds to read.
SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46, 47)
DRY_SEEDS: Tuple[int, ...] = (42, 43)

RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# Pre-registered thresholds (constants, NOT derived from the run's own statistics).
RESISTANCE_DOSE_MARGIN = 0.15          # retained-fraction growth min-to-max dose for SUPPORTED
INSTALL_TOOK_MAJORITY = 0.5            # >50% of an arm's seeds must have install_took for scorable
MIN_SCORABLE_ARMS = 2                  # fewer than 2 took -> dose-response unmeasurable

def _arm_id(dose: int) -> str:
    return f"dose_bc{int(dose)}"


# --- Per-arm readiness precondition (install must TAKE, or the arm is uninformative) --------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="install_took_strict_majority",
        description=(
            "A strict majority of this dose arm's seeds installed above the competence floor "
            "(post_bc >= COMPETENCE_RESOURCE_FLOOR). An install that did not take is "
            "UNINFORMATIVE about retention; below this the arm reds ONLY ITSELF and never vacates "
            "a sibling dose arm's valid result."
        ),
        control="fraction of seeds with install_took=True in this dose arm, vs 0.5",
        threshold=float(INSTALL_TOOK_MAJORITY),
        direction="lower",   # a FLOOR: met when measured > threshold
        kind="readiness",
    ),
)
PRECONDITION_BY_NAME: Dict[str, PreconditionSpec] = {s.name: s for s in PRECONDITION_SPECS}


def _arm_contexts(doses: Tuple[int, ...]) -> List[Dict[str, Any]]:
    return [{"id": _arm_id(d), "dose": int(d)} for d in doses]


def _run_cell(dose: int, seed: int, env_kwargs: Dict[str, Any], *,
              on_budget: int, eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    """One (dose x seed) cell: BC-install at `dose` episodes, then the SAME unconstrained RL
    refinement, probed on a cadence. Assembles the trajectory-shaped retention row."""
    arm_id = _arm_id(dose)
    print(f"Seed {seed} Condition {arm_id}", flush=True)   # boundary reset for the runner

    # The unconstrained reference RL refinement -- identical for every dose arm.
    cfg = baselines.reference_config(on_budget, retention_probe_every=int(probe_every))

    # The declared config_slice: what THIS cell's computation reads. The install dose is part of
    # it (a different dose is a different computation). Leg-specific hash (driver folded in) --
    # not minted for cross-driver reuse, so a false HIT is impossible.
    slice_cfg: Dict[str, Any] = {
        "arm_id": arm_id,
        "rung_id": fan.RUNG_ID,
        "kind": "mech476_dose_consolidation",
        "bc_install_dose": int(dose),
        "env_kwargs": dict(env_kwargs),
        "representation": baselines.REF_REPRESENTATION,
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
        "retention_probe_every": int(probe_every),
    }
    slice_cfg.update(cfg.as_slice())

    with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=True) as cell:
        rep_agent = baselines.build_off_arm(seed, env_kwargs, steps=int(steps), cfg=cfg)

        # Phase 1 -- BC install at THIS dose, then measure whether it TOOK.
        install = baselines.install_bc_prior(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id,
            n_bc=int(dose),
        )
        post_bc = float(install["post_bc_foraging_competence"])

        # Phase 2 -- the SAME unconstrained RL refinement, probed on a cadence.
        probe_fn = baselines.make_probe_fn(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
        )
        guard = baselines.train_off_arm(
            rep_agent, seed, env_kwargs, steps=steps, arm_label=arm_id, cfg=cfg,
            demo=install["demo"], probe_fn=probe_fn,
        )
        trajectory: List[Dict[str, Any]] = list(guard.get("competence_trajectory", []))

        # Phase 3 -- unshaped terminal eval (same statistic as the probe).
        eval_env = x734._make_env(seed, env_kwargs)
        row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)

        traj_vals = [float(r.get("foraging_competence", 0.0)) for r in trajectory]
        peak = round(max(traj_vals), 6) if traj_vals else 0.0
        terminal_traj = round(traj_vals[-1], 6) if traj_vals else 0.0
        retained = baselines.retained_fraction(trajectory, post_bc)
        half_life = baselines.competence_half_life(trajectory, post_bc)

        row["rung_id"] = fan.RUNG_ID
        row["arm_id"] = arm_id
        row["seed"] = int(seed)
        row["bc_install_dose"] = int(install.get("bc_install_dose", dose))
        row["post_bc_foraging_competence"] = round(post_bc, 6)
        row["install_took"] = bool(install["install_took"])
        row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
        row["competence_trajectory"] = trajectory        # FULL per-seed trajectory, never collapsed
        row["n_trajectory_readings"] = int(len(trajectory))
        row["trajectory_peak_competence"] = peak
        row["trajectory_terminal_competence"] = terminal_traj
        row["retained_fraction"] = retained
        row["competence_half_life_episodes"] = half_life
        row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
        row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
        row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
        cell.stamp(row)

    forage = float(row["foraging_competence"])
    print(
        f"verdict: {'PASS' if row.get('competence_supra_floor', forage > COMPETENCE_RESOURCE_FLOOR) else 'FAIL'} "
        f"(arm={arm_id} seed={seed} dose={dose} post_bc={post_bc} "
        f"retained_fraction={retained} forage/ep={forage})", flush=True,
    )
    return row


def run_experiment(seeds: Tuple[int, ...], doses: Tuple[int, ...], *,
                   on_budget: int, eval_eps: int, steps: int, probe_every: int) -> Dict[str, Any]:
    print(
        f"MECH-476 DOSE-dependent consolidation falsifier "
        f"({len(doses)} dose arms x {len(seeds)} seeds; doses={list(doses)}; "
        f"rep={baselines.REF_REPRESENTATION}, RL_budget={on_budget}, probe_every={probe_every}, "
        f"eval={eval_eps}, steps={steps}; manipulation=BC install dose ONLY, "
        f"refinement=unconstrained reference)",
        flush=True,
    )
    arm_ctxs = _arm_contexts(doses)
    # Design-audit BEFORE any compute: the readiness precondition carries no structural bound, so
    # this refuses nothing here -- it is the same guard 792a/785 use and is kept for uniformity.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    all_cells: List[Dict[str, Any]] = []
    for dose in doses:
        for seed in seeds:
            all_cells.append(_run_cell(dose, seed, env_kwargs, on_budget=on_budget,
                                       eval_eps=eval_eps, steps=steps, probe_every=probe_every))

    # ---- per-arm retention readouts (trajectory-shaped) -----------------------------------
    def _cells_for(dose: int) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == _arm_id(dose)]

    per_arm: Dict[str, Any] = {}
    arm_gates: List[Dict[str, Any]] = []
    for dose in doses:
        arm_id = _arm_id(dose)
        cells = _cells_for(dose)
        n_cells = len(cells)
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        install_took_frac = float(n_install_took / n_cells) if n_cells else 0.0
        # retained_fraction only over cells whose install TOOK (a None or un-taken install is
        # NOT a maximal-erosion reading -- it is uninformative).
        retained_took = [
            float(c["retained_fraction"]) for c in cells
            if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None
        ]
        retained_mean = round(sum(retained_took) / len(retained_took), 6) if retained_took else None
        post_bc_vals = [float(c.get("post_bc_foraging_competence", 0.0)) for c in cells]
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        terminals = [float(c.get("trajectory_terminal_competence", 0.0)) for c in cells]

        # per-arm readiness gate (install_took strict majority). A red arm never vacates a sibling.
        measured = {"install_took_strict_majority": install_took_frac}
        gate = evaluate_arm_gate(arm_id, {"id": arm_id, "dose": int(dose)},
                                 PRECONDITION_SPECS, measured=measured)
        arm_gates.append(gate)

        per_arm[arm_id] = {
            "arm_id": arm_id,
            "bc_install_dose": int(dose),
            "n_cells": n_cells,
            "n_seeds_install_took": n_install_took,
            "install_took_fraction": round(install_took_frac, 6),
            "post_bc_foraging_competence_per_seed": [round(v, 6) for v in post_bc_vals],
            "post_bc_foraging_competence_mean": round(sum(post_bc_vals) / n_cells, 6) if n_cells else 0.0,
            "retained_fraction_per_seed": [c.get("retained_fraction") for c in cells],
            "retained_fraction_mean_over_took": retained_mean,
            "n_seeds_retained_fraction_defined": len(retained_took),
            "trajectory_peak_mean": round(sum(peaks) / n_cells, 6) if n_cells else 0.0,
            "trajectory_terminal_mean": round(sum(terminals) / n_cells, 6) if n_cells else 0.0,
            "gate_green": bool(gate["gate_green"]),
        }

    gate_agg = aggregate_arm_gates(arm_gates)
    green_arms = list(gate_agg["per_arm_gate"]["green_arms"])       # arms whose install took
    n_scorable = len(green_arms)

    # ---- dose-response scoring over the SCORABLE (install-took) arms ------------------------
    # Ordered by dose so monotonicity is read low->high.
    scorable_by_dose = sorted(
        [(int(per_arm[a]["bc_install_dose"]), per_arm[a]["retained_fraction_mean_over_took"])
         for a in green_arms
         if per_arm[a]["retained_fraction_mean_over_took"] is not None],
        key=lambda t: t[0],
    )
    dose_axis = [d for d, _ in scorable_by_dose]
    retained_axis = [m for _, m in scorable_by_dose]

    # Distinguish UNMEASURABLE (degenerate) from a MEASURED invariance (a real WEAKENED verdict).
    measurable = len(scorable_by_dose) >= MIN_SCORABLE_ARMS
    non_degenerate = bool(measurable)
    degeneracy_reason = "" if measurable else (
        f"fewer than {MIN_SCORABLE_ARMS} dose arms achieved majority install_took with a defined "
        f"retained_fraction ({len(scorable_by_dose)} scorable) -> dose-response UNMEASURABLE; "
        f"substrate_not_ready_requeue, NOT a retention verdict"
    )

    spread = None
    monotone = None
    supported = weakened = non_monotone = False
    if measurable:
        spread = round(max(retained_axis) - min(retained_axis), 6)
        # monotone non-decreasing in dose
        monotone = all(retained_axis[i] <= retained_axis[i + 1] + 1e-9
                       for i in range(len(retained_axis) - 1))
        grows = bool((retained_axis[-1] - retained_axis[0]) >= RESISTANCE_DOSE_MARGIN)
        if monotone and grows:
            supported = True
        elif spread < RESISTANCE_DOSE_MARGIN:
            weakened = True
        else:
            non_monotone = True

    # ---- interpretation + outcome ----------------------------------------------------------
    if not measurable:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
    elif supported:
        label = "consolidation_process_dose_dependent"
        outcome = "PASS"
        evidence_direction = "supports"
    elif weakened:
        label = "retention_invariant_to_dose_no_process"
        outcome = "FAIL"
        evidence_direction = "weakens"
    else:
        label = "non_monotone_dose_response"
        outcome = "FAIL"
        evidence_direction = "mixed"

    # criteria_non_degenerate, keyed to each arm's gate (the channel the indexer honours).
    criteria_by_arm = {a: ["retained_fraction_measured"] for a in (_arm_id(d) for d in doses)}
    criteria_nd = arm_criteria_non_degenerate(criteria_by_arm, gate_agg["per_arm_gate"])

    interpretation = {
        "label": label,
        "preconditions": gate_agg["adjudication_preconditions"],
        "preconditions_scope_note": gate_agg["per_arm_gate"]["preconditions_scope_note"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {"name": "resistance_grows_with_dose", "load_bearing": True,
             "passed": bool(supported)},
        ],
    }

    return {
        "outcome": outcome,
        "interpretation_label": label,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "measurable": measurable,
        "n_scorable_arms": n_scorable,
        "dose_response": {
            "dose_axis": dose_axis,
            "retained_fraction_mean_axis": retained_axis,
            "spread": spread,
            "monotone_non_decreasing": monotone,
            "resistance_dose_margin": RESISTANCE_DOSE_MARGIN,
            "supported": supported,
            "weakened": weakened,
            "non_monotone": non_monotone,
        },
        "per_arm": per_arm,
        "per_arm_gate": gate_agg["per_arm_gate"],
        "interpretation": interpretation,
        "all_cells": all_cells,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, *, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "timestamp_utc": timestamp_utc,
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "dose_response": result["dose_response"],
        "per_arm": result["per_arm"],
        "per_arm_gate": result["per_arm_gate"],
        "arm_results": result["all_cells"],
        "sleep_driver_pattern": "none",
        "dry_run": bool(dry_run),
        # pre-registered constants + provenance
        "doses_bc_episodes": list(DOSES),
        "resistance_dose_margin": RESISTANCE_DOSE_MARGIN,
        "install_took_majority": INSTALL_TOOK_MAJORITY,
        "min_scorable_arms": MIN_SCORABLE_ARMS,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "retention_probe_every": RETENTION_PROBE_EVERY,
        "claim_under_test": "MECH-476 (dose axis of the A->B->A retrograde-interference falsifier)",
        "blocked_arms_routed_to_implement_substrate": (
            "A->B offline INTERVAL (consolidation process) and Moncada-2007 NOVELTY-tagging "
            "pairing -- both need an offline policy-consolidation window the substrate lacks "
            "(SLEEP DRIVER: none). Home: sleep_substrate_plan.md SD-017 / MECH-204."
        ),
        "reference_build": "128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32, "
                           "unconstrained (no KL anchor, no distributional critic)",
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--doses", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="RL refinement budget per cell")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = tuple(args.seeds) if args.seeds else DRY_SEEDS
        doses = tuple(args.doses) if args.doses else DRY_DOSES
        on_budget = args.episodes if args.episodes is not None else fan.DRY_RL
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 3
        steps = args.steps if args.steps is not None else 40
        probe_every = args.probe_every if args.probe_every is not None else DRY_RETENTION_PROBE_EVERY
    else:
        seeds = tuple(args.seeds) if args.seeds else SEEDS
        doses = tuple(args.doses) if args.doses else DOSES
        on_budget = args.episodes if args.episodes is not None else int(
            fan.RL_EPISODES * baselines.REF_BUDGET_MULTIPLIER)   # 1000 * 3 = 3000
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 20
        steps = args.steps if args.steps is not None else 60
        probe_every = args.probe_every if args.probe_every is not None else RETENTION_PROBE_EVERY

    result = run_experiment(seeds, doses, on_budget=on_budget, eval_eps=eval_eps,
                            steps=steps, probe_every=probe_every)

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cfg: Dict[str, Any] = {
        "doses": list(doses),
        "seeds": list(seeds),
        "rl_budget_per_cell": int(on_budget),
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
        "retention_probe_every": int(probe_every),
        "representation": baselines.REF_REPRESENTATION,
        "reference_budget_multiplier": baselines.REF_BUDGET_MULTIPLIER,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    # AFTER arm_results is assembled, so substrate_hash hoists from the per-cell fingerprints.
    stamp_recording_core(manifest, config=cfg, seeds=list(seeds), script_path=Path(__file__),
                         started_at=t0)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=list(seeds),
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    dr = result["dose_response"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"evidence={result['evidence_direction']} non_degenerate={result['non_degenerate']} "
        f"scorable_arms={result['n_scorable_arms']}", flush=True,
    )
    for dose in doses:
        r = result["per_arm"][_arm_id(dose)]
        print(
            f"  {r['arm_id']}: dose={r['bc_install_dose']} "
            f"install_took={r['n_seeds_install_took']}/{r['n_cells']} "
            f"post_bc_mean={r['post_bc_foraging_competence_mean']} "
            f"retained_frac_mean={r['retained_fraction_mean_over_took']} "
            f"green={r['gate_green']} peak_mean={r['trajectory_peak_mean']}", flush=True,
        )
    print(
        f"  dose_response: doses={dr['dose_axis']} retained={dr['retained_fraction_mean_axis']} "
        f"spread={dr['spread']} monotone={dr['monotone_non_decreasing']} "
        f"(margin={dr['resistance_dose_margin']}) "
        f"supported={dr['supported']} weakened={dr['weakened']} non_monotone={dr['non_monotone']}",
        flush=True,
    )
    print(
        f"  green_arms={result['per_arm_gate']['green_arms']} "
        f"red_arms={result['per_arm_gate']['red_arms']}", flush=True,
    )
    if result["degeneracy_reason"]:
        print(f"  degeneracy_reason: {result['degeneracy_reason']}", flush=True)

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
