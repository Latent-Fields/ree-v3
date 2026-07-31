#!/opt/local/bin/python3
"""V3-EXQ-836d -- MECH-476 falsifier (NOVELTY axis / Moncada-2007 behavioural tagging, REDESIGN):
does a WEAK (sub-threshold) install consolidate ONLY when a novelty episode is paired with it
inside the offline window? Same question as V3-EXQ-836c; the DISCRIMINATION RULE is corrected,
not the design.

SUPERSEDES V3-EXQ-836c. WHY. failure_autopsy_mech476-mech475-cluster_2026-07-29.md diagnosed
836c's FAIL (reversed_novelty_effect) as a measurement/test-design defect, not a genuine reversal:
the SUPPORTED/WEAKENED/REVERSED call rested on a FIXED literal novelty_retention_margin=0.15, not
scaled to observed per-arm noise, and the entire reversal is attributable to ONE outlier seed in
the unpaired arm (retained_fraction=2.7671, 2-15x every sibling in the same arm; arm SD 0.9026 vs
0.3347 for the paired arm). Removing that single seed flips the sign back to (weakly) SUPPORT the
paired-retains-more prediction (paired - unpaired = +0.0383), well inside the margin either way --
the result does not survive a leave-one-out check. Per the project's own documented effect-size-
gate convention (scale noise on the SD of the delta between arms, never a bare fixed number -- see
v3_exq_824a_q081_shared_organisation_landmark_removal.py:EFFECT_SIZE_K/EFFECT_SIZE_ABS_FLOOR),
836c's discrimination rule violated that convention. This redesign corrects it and re-runs the
SAME design at higher power. User-confirmed routing via AskUserQuestion 2026-07-29 ("Redesign
re-run (noise-scaled margin + more seeds)"); hypothesis_space_registry.v1.json qid=competence_floor
hid=H-mech476-novelty-tagging left `alive` (not eliminated -- 836c was uninformative, not
falsifying).

WHAT CHANGED FROM V3-EXQ-836c (repair pathway, both applied; autopsy said "and/or"):
  1. NOVELTY_RETENTION_MARGIN (a fixed 0.15 literal) -> `effective_novelty_margin`, a noise-scaled
     floor `max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR)` computed from the SD of the
     per-seed PAIRED delta between the paired and unpaired arms (same seed integer -> same RNG
     seed at cell construction in both arms, so a per-seed pairing is available exactly as in the
     original design's shared SEEDS tuple). Mirrors the 824a convention verbatim.
  2. SEEDS: 6 -> 10 per arm (n>=10, per the autopsy's explicit floor). This alone would have
     diluted 836c's single outlier's leverage on the unpaired-arm mean from 1/6 to 1/10, and the
     noise-scaled floor (rather than a fixed 0.15) would also have widened the required margin in
     response to that arm's own higher observed spread.
  3. Robust statistic: `retained_fraction_median_over_took` reported per arm ALONGSIDE the mean
     (never replacing it), so a reader can see whether the mean is outlier-driven -- exactly the
     defect this autopsy diagnosed.
  4. Leave-one-out robustness diagnostic (`leave_one_out` block): for the paired/unpaired paired
     delta, recompute BOTH the mean delta and the noise-scaled floor with each single seed dropped
     in turn, and record whether the three-way verdict (supported/weakened/reversed) is STABLE
     across every fold. This is the exact check the autopsy applied by hand, now run automatically
     and pre-registered so the verdict is falsifiable rather than asserted.

CLAIM UNDER TEST (unchanged from 836c). MECH-476 (candidate / v3_pending, split_from MECH-457).
Its third, sharpest arm, from the behavioural-tagging literature (Moncada & Viola 2007, J Neurosci
27:7476): a WEAK training that would induce only short-term memory CONSOLIDATES INTO LONG-TERM
memory when paired with exploration of a NOVEL environment close in time. A regulariser has NO
reason to care about an unrelated novelty exposure, so a novelty-gated dissociation is evidence for
a genuine consolidation process rather than concurrent regularisation.

SUBSTRATE (unchanged). The offline policy-consolidation window (SD-083, ree-v3 42ab95f688,
experiments/_lib/mech457_offline_consolidation.py). Its capture resource is NOVELTY-GATED: the
lineage's own RNDModule (RepAgent feature space -- the MECH-441 novelty principle for this stack)
measures the novelty of an exposure run inside the window, and a weak tag is captured only when the
paired condition supplies novelty. offline_novelty_pairing: "paired" -> novelty_factor = measured
novelty (clamped to [tag_leak, 1]); "unpaired" -> novelty_factor pinned at the tag_leak floor.

WHAT THIS LEG TESTS (unchanged). Novelty PAIRING, holding a WEAK-BUT-TOOK install dose and the
offline INTERVAL FIXED. Two arms differ ONLY in offline_novelty_pairing:
  * novelty_paired   -- a novelty exposure runs inside the window; the weak tag is captured.
  * novelty_unpaired -- no novelty pairing; the weak tag decays (capture at the tag_leak floor).

The install is deliberately WEAK ("sub-threshold"): FIXED_WEAK_BC_DOSE=150 BC episodes -- low
enough that the unconstrained RL refinement should erode it, but (guarded by the install_took
precondition) above COMPETENCE_RESOURCE_FLOOR so retention is measurable. A weak install that did
not take is UNINFORMATIVE and self-routes substrate_not_ready_requeue.

DESIGN (unchanged). Two arms, both BC-install the raw_view policy at the WEAK dose
(baselines.install_bc_prior(n_bc=150)), then run the SD-083 offline window at the SAME fixed
interval N=FIXED_WINDOW_STEPS with opposite novelty pairing, then the IDENTICAL unconstrained
reference RL refinement under the window's EWC anchor. The ONLY thing that varies is the novelty
pairing.

DV -- a TRAJECTORY (unchanged). retained_fraction = terminal_trajectory_competence /
post_bc_installed_competence (baselines.retained_fraction; None when install ~0).

DV-SYMMETRY (mandatory, per arm; unchanged). The DV is retained_fraction (a competence ratio). The
manipulation is offline_novelty_pairing. retained_fraction is NOT invariant under
paired<->unpaired: the pairing sets the window's novelty_factor (measured novelty for paired vs
the tag_leak floor for unpaired), which scales the effective EWC coefficient, which changes the
trace-selective protection applied during RL and hence the erosion. So the manipulation is not a
symmetry of the DV, for either arm. This is a genuine measurement, not an arithmetic identity.

SCORING (pre-registered, discrimination rule CORRECTED).
  * SUPPORTED (PASS, supports MECH-476 / behavioural tagging): over arms whose install TOOK,
    retained_fraction_mean[paired] - retained_fraction_mean[unpaired] >= effective_novelty_margin,
    where effective_novelty_margin = max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR) and
    sd_delta is the SD of the per-seed paired (paired - unpaired) retained_fraction delta. The
    weak install consolidates ONLY when novelty is paired -> a genuine, novelty-gated
    consolidation process.
  * WEAKENED (FAIL, weakens MECH-476): |paired - unpaired| < effective_novelty_margin. Novelty
    makes no difference -> a regulariser, not a tagging process; the intended failure mode.
  * REVERSED (FAIL, mixed): paired retains LESS than unpaired by >= effective_novelty_margin --
    anomalous; routes to /failure-autopsy rather than a clean verdict.

PREREQUISITE (NON-NEGOTIABLE, unchanged). Each arm carries an install_took strict-majority
readiness precondition (both arms share the SAME weak dose, so install_took should be uniform). A
red arm reds ONLY ITSELF; if FEWER THAN TWO arms take (i.e. either arm), the novelty contrast is
UNMEASURABLE and the run self-routes substrate_not_ready_requeue, NOT a WEAKENED verdict.
Additionally, a paired arm whose measured novelty is at the floor (a mislabelled non-novel
exposure) is flagged -- the offline window's own novelty measurement is the positive control that
the pairing manipulation actually fired.

CALIBRATION NOTE (pre-registered, auditable, unchanged). The weak dose (150) and the
pre-registered margin are chosen so the UNPAIRED arm erodes while the PAIRED arm is protected. If
the first real result shows the unpaired arm ALSO fully retaining (dose too strong -> no room for
novelty to matter) or BOTH arms at the floor (dose too weak -> nothing took), the correct route is
a re-queue at a re-calibrated dose (a lettered iteration), NOT a WEAKENED verdict. The
install_took gate catches the too-weak case automatically; the too-strong case is read off
retained_fraction_mean[unpaired] ~ 1.0.

RE-DERIVE BRAKE / GOV-REUSE-1: this is a same-question, corrected-instrument re-run of a run whose
own autopsy scored `epistemic_category: measurement_test_design_defect` (NOT `substrate_ceiling`)
with `recommended_substrate_queue_entry.action: none` -- the re-derive brake does not apply, and
the decisive readout (a noise-scaled novelty-contrast verdict) is NOT recorded anywhere: 836c's
own manifest computed it under the wrong rule and at n=6, which the redesign's n=10 requires
re-running to supply. Must re-run.

EXPERIMENT_PURPOSE = evidence. Directly tests MECH-476's behavioural-tagging prediction. Single
claim.

ethics_preflight: all-false / allow. No negative valence, no suffering-like state, no self-model,
no offline replay over harm, no social mind / language, no human/clinical data. V3 pre-ethical
instrumentation only (SENT-0). The novelty exposure is exploration of a fresh grid region
(appetitive foraging), not an aversive manipulation; the window replays REAL on-expert states
(MECH-094 N/A).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF). The SD-083
offline window is a POLICY-consolidation mechanism in the mech457 testbed, distinct from the
cognifold SD-017 sleep loop.
"""

from __future__ import annotations

import argparse
import statistics
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

EXPERIMENT_TYPE = "v3_exq_836d_mech476_novelty_tagging_consolidation_redesign"
QUEUE_ID = "V3-EXQ-836d"
SUPERSEDES = "V3-EXQ-836c"
CLAIM_IDS: List[str] = ["MECH-476"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Novelty-pairing contrast (the ONLY manipulation) --------------------------------------
PAIRINGS: Tuple[str, ...] = ("paired", "unpaired")

# Held FIXED across both arms.
FIXED_WEAK_BC_DOSE = 150               # sub-threshold: erodable but (guarded) above the floor
FIXED_WINDOW_STEPS = 300               # offline interval, held fixed (capture ~0.78 at tau=200)
OFFLINE_EWC_MAX_COEF = 100.0           # pre-registered; scaled by capture x novelty_factor
OFFLINE_CAPTURE_TAU = 200.0
OFFLINE_CAPTURE_MAX = 1.0
OFFLINE_FISHER_SAMPLES = 256
DRY_FISHER_SAMPLES = 8

# REDESIGN: n=6 -> n=10 (autopsy-mandated floor). Same seed integers reused across both arms so a
# per-seed PAIRED delta is available for the noise-scaled floor.
SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
DRY_SEEDS: Tuple[int, ...] = (42, 43)

RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2

# Pre-registered thresholds (constants, NOT derived from the run's own statistics). The FLOOR
# itself is now noise-scaled (effective_novelty_margin, computed in run_experiment); K and
# ABS_FLOOR are the pre-registered scaling constants, mirroring v3_exq_824a's
# EFFECT_SIZE_K/ABS_FLOOR convention exactly. Same K/ABS_FLOOR as the sibling 836a redesign
# (same DV family, same noise regime) for a consistent discrimination standard across MECH-476's
# arms.
EFFECT_SIZE_K = 1.5
EFFECT_SIZE_ABS_FLOOR = 0.05
INSTALL_TOOK_MAJORITY = 0.5
MIN_SCORABLE_ARMS = 2                  # both arms must take for the contrast to be measurable


def _arm_id(pairing: str) -> str:
    return f"novelty_{pairing}"


PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="install_took_strict_majority",
        description=(
            "A strict majority of this arm's seeds installed above the competence floor "
            "(post_bc >= COMPETENCE_RESOURCE_FLOOR). A weak install that did not take is "
            "UNINFORMATIVE about retention; below this the arm reds ONLY ITSELF."
        ),
        control="fraction of seeds with install_took=True in this arm, vs 0.5",
        threshold=float(INSTALL_TOOK_MAJORITY),
        direction="lower",
        kind="readiness",
    ),
)


def _arm_contexts(pairings: Tuple[str, ...]) -> List[Dict[str, Any]]:
    return [{"id": _arm_id(p), "pairing": p} for p in pairings]


def _run_cell(pairing: str, seed: int, env_kwargs: Dict[str, Any], *,
              on_budget: int, eval_eps: int, steps: int, probe_every: int,
              fisher_samples: int) -> Dict[str, Any]:
    """One (pairing x seed) cell: weak BC-install, SD-083 offline window at the fixed interval with
    this arm's novelty pairing, then the SAME unconstrained RL refinement under the EWC anchor."""
    arm_id = _arm_id(pairing)
    print(f"Seed {seed} Condition {arm_id}", flush=True)   # boundary reset for the runner

    cfg = baselines.reference_config(
        on_budget, retention_probe_every=int(probe_every),
        use_offline_consolidation=True,
        offline_window_steps=int(FIXED_WINDOW_STEPS),
        offline_capture_tau=OFFLINE_CAPTURE_TAU,
        offline_capture_max=OFFLINE_CAPTURE_MAX,
        offline_ewc_max_coef=OFFLINE_EWC_MAX_COEF,
        offline_fisher_samples=int(fisher_samples),
        offline_novelty_pairing=str(pairing),     # THE manipulation
    )

    slice_cfg: Dict[str, Any] = {
        "arm_id": arm_id,
        "rung_id": fan.RUNG_ID,
        "kind": "mech476_novelty_tagging_redesign",
        "bc_install_dose": int(FIXED_WEAK_BC_DOSE),
        "offline_window_steps": int(FIXED_WINDOW_STEPS),
        "offline_novelty_pairing": str(pairing),
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

        # Phase 1 -- WEAK BC install, then measure whether it TOOK.
        install = baselines.install_bc_prior(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id,
            n_bc=int(FIXED_WEAK_BC_DOSE),
        )
        post_bc = float(install["post_bc_foraging_competence"])

        # Phase 1.5 -- SD-083 offline window at the fixed interval with THIS arm's novelty pairing.
        window = baselines.consolidate_offline_window(
            rep_agent, seed, env_kwargs, steps=steps, cfg=cfg, demo=install["demo"],
        )
        anchor = window.get("anchor")

        # Phase 2 -- the SAME unconstrained RL refinement under the offline EWC anchor, probed.
        probe_fn = baselines.make_probe_fn(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
        )
        guard = baselines.train_off_arm(
            rep_agent, seed, env_kwargs, steps=steps, arm_label=arm_id, cfg=cfg,
            demo=install["demo"], probe_fn=probe_fn, offline_ewc_anchor=anchor,
        )
        trajectory: List[Dict[str, Any]] = list(guard.get("competence_trajectory", []))

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
        row["offline_novelty_pairing"] = str(pairing)
        row["offline_window_steps"] = int(FIXED_WINDOW_STEPS)
        row["bc_install_dose"] = int(install.get("bc_install_dose", FIXED_WEAK_BC_DOSE))
        row["post_bc_foraging_competence"] = round(post_bc, 6)
        row["install_took"] = bool(install["install_took"])
        row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
        row["offline_capture_resource"] = float(window.get("capture_resource", 0.0))
        row["offline_novelty_factor"] = float(window.get("novelty_factor", 0.0))
        row["offline_measured_novelty"] = window.get("measured_novelty")   # None for unpaired
        row["offline_effective_ewc_coef"] = float(window.get("effective_ewc_coef", 0.0))
        row["offline_fisher_mass"] = float(window.get("fisher_mass", 0.0))
        row["offline_n_fisher_states"] = int(window.get("n_fisher_states", 0))
        row["offline_ewc_installed"] = bool(guard.get("offline_ewc_installed", False))
        row["mean_offline_ewc_penalty_recent"] = float(guard.get("mean_offline_ewc_penalty_recent", 0.0))
        row["competence_trajectory"] = trajectory
        row["n_trajectory_readings"] = int(len(trajectory))
        row["trajectory_peak_competence"] = peak
        row["trajectory_terminal_competence"] = terminal_traj
        row["retained_fraction"] = retained
        row["competence_half_life_episodes"] = half_life
        row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
        row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
        cell.stamp(row)

    forage = float(row["foraging_competence"])
    print(
        f"verdict: {'PASS' if row.get('competence_supra_floor', forage > COMPETENCE_RESOURCE_FLOOR) else 'FAIL'} "
        f"(arm={arm_id} seed={seed} pairing={pairing} post_bc={post_bc} "
        f"novelty_factor={row['offline_novelty_factor']:.3f} ewc_coef={row['offline_effective_ewc_coef']:.3f} "
        f"retained_fraction={retained} forage/ep={forage})", flush=True,
    )
    return row


def _paired_deltas(paired_cells: List[Dict[str, Any]],
                    unpaired_cells: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Per-seed (paired - unpaired) retained_fraction delta, matched by seed integer. Only seeds
    where BOTH arms installed (install_took) and have a defined retained_fraction contribute."""
    by_seed_paired = {int(c["seed"]): c for c in paired_cells
                      if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None}
    by_seed_unpaired = {int(c["seed"]): c for c in unpaired_cells
                        if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None}
    shared = sorted(set(by_seed_paired) & set(by_seed_unpaired))
    return [(s, float(by_seed_paired[s]["retained_fraction"]) - float(by_seed_unpaired[s]["retained_fraction"]))
            for s in shared]


def _novelty_verdict(mean_delta: float, effective_floor: float) -> str:
    """The pre-registered three-way discrimination, factored so the leave-one-out diagnostic can
    call it identically on each fold. Returns one of 'supported' / 'weakened' / 'reversed'."""
    if mean_delta >= effective_floor:
        return "supported"
    if mean_delta <= -effective_floor:
        return "reversed"
    return "weakened"


def run_experiment(seeds: Tuple[int, ...], pairings: Tuple[str, ...], *,
                   on_budget: int, eval_eps: int, steps: int, probe_every: int,
                   fisher_samples: int) -> Dict[str, Any]:
    print(
        f"MECH-476 NOVELTY-tagging consolidation falsifier REDESIGN "
        f"({len(pairings)} pairing arms x {len(seeds)} seeds; pairings={list(pairings)}; "
        f"rep={baselines.REF_REPRESENTATION}, weak_dose={FIXED_WEAK_BC_DOSE}, N={FIXED_WINDOW_STEPS}, "
        f"RL_budget={on_budget}, probe_every={probe_every}, eval={eval_eps}, steps={steps}; "
        f"ewc_max_coef={OFFLINE_EWC_MAX_COEF}, fisher_samples={fisher_samples}; "
        f"manipulation=novelty pairing ONLY, refinement=unconstrained reference; "
        f"discrimination_margin=noise-scaled max({EFFECT_SIZE_K}*sd_delta, {EFFECT_SIZE_ABS_FLOOR}))",
        flush=True,
    )
    arm_ctxs = _arm_contexts(pairings)
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    all_cells: List[Dict[str, Any]] = []
    for pairing in pairings:
        for seed in seeds:
            all_cells.append(_run_cell(pairing, seed, env_kwargs, on_budget=on_budget,
                                       eval_eps=eval_eps, steps=steps, probe_every=probe_every,
                                       fisher_samples=fisher_samples))

    def _cells_for(pairing: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == _arm_id(pairing)]

    per_arm: Dict[str, Any] = {}
    arm_gates: List[Dict[str, Any]] = []
    for pairing in pairings:
        arm_id = _arm_id(pairing)
        cells = _cells_for(pairing)
        n_cells = len(cells)
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        install_took_frac = float(n_install_took / n_cells) if n_cells else 0.0
        retained_took = [
            float(c["retained_fraction"]) for c in cells
            if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None
        ]
        retained_mean = round(sum(retained_took) / len(retained_took), 6) if retained_took else None
        # REDESIGN: robust statistic reported ALONGSIDE the mean -- exactly the read that would
        # have flagged 836c's single-seed-driven reversal at authoring time.
        retained_median = round(statistics.median(retained_took), 6) if retained_took else None
        post_bc_vals = [float(c.get("post_bc_foraging_competence", 0.0)) for c in cells]
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        terminals = [float(c.get("trajectory_terminal_competence", 0.0)) for c in cells]
        nov_factor_vals = [float(c.get("offline_novelty_factor", 0.0)) for c in cells]
        measured_nov = [c.get("offline_measured_novelty") for c in cells
                        if c.get("offline_measured_novelty") is not None]
        ewc_coef_vals = [float(c.get("offline_effective_ewc_coef", 0.0)) for c in cells]

        measured = {"install_took_strict_majority": install_took_frac}
        gate = evaluate_arm_gate(arm_id, {"id": arm_id, "pairing": pairing},
                                 PRECONDITION_SPECS, measured=measured)
        arm_gates.append(gate)

        per_arm[arm_id] = {
            "arm_id": arm_id,
            "offline_novelty_pairing": pairing,
            "n_cells": n_cells,
            "n_seeds_install_took": n_install_took,
            "install_took_fraction": round(install_took_frac, 6),
            "post_bc_foraging_competence_per_seed": [round(v, 6) for v in post_bc_vals],
            "post_bc_foraging_competence_mean": round(sum(post_bc_vals) / n_cells, 6) if n_cells else 0.0,
            "retained_fraction_per_seed": [c.get("retained_fraction") for c in cells],
            "retained_fraction_mean_over_took": retained_mean,
            "retained_fraction_median_over_took": retained_median,
            "n_seeds_retained_fraction_defined": len(retained_took),
            "trajectory_peak_mean": round(sum(peaks) / n_cells, 6) if n_cells else 0.0,
            "trajectory_terminal_mean": round(sum(terminals) / n_cells, 6) if n_cells else 0.0,
            "offline_novelty_factor_mean": round(sum(nov_factor_vals) / n_cells, 6) if n_cells else 0.0,
            "offline_measured_novelty_mean": (round(sum(measured_nov) / len(measured_nov), 6)
                                              if measured_nov else None),
            "offline_effective_ewc_coef_mean": round(sum(ewc_coef_vals) / n_cells, 6) if n_cells else 0.0,
            "gate_green": bool(gate["gate_green"]),
        }

    gate_agg = aggregate_arm_gates(arm_gates)
    green_arms = list(gate_agg["per_arm_gate"]["green_arms"])
    n_scorable = len(green_arms)

    # ---- novelty-contrast scoring over the SCORABLE (install-took) arms -------------------
    paired_id, unpaired_id = _arm_id("paired"), _arm_id("unpaired")
    paired_mean = per_arm.get(paired_id, {}).get("retained_fraction_mean_over_took")
    unpaired_mean = per_arm.get(unpaired_id, {}).get("retained_fraction_mean_over_took")
    both_scorable = (paired_id in green_arms and unpaired_id in green_arms
                     and paired_mean is not None and unpaired_mean is not None)

    measurable = bool(both_scorable) and (n_scorable >= MIN_SCORABLE_ARMS)
    non_degenerate = bool(measurable)
    degeneracy_reason = "" if measurable else (
        "both novelty arms must achieve majority install_took with a defined retained_fraction for "
        "the paired-vs-unpaired contrast to be measurable -> novelty contrast UNMEASURABLE; "
        "substrate_not_ready_requeue, NOT a retention verdict"
    )

    delta = None
    supported = weakened = reversed_effect = False
    sd_delta = None
    effective_floor = None
    leave_one_out: Dict[str, Any] = {"applicable": False}
    if measurable:
        # REDESIGN: noise-scaled floor, computed from the per-seed PAIRED delta between the
        # paired and unpaired arms (the comparison the load-bearing criterion reads).
        deltas = _paired_deltas(_cells_for("paired"), _cells_for("unpaired"))
        delta_vals = [d for _, d in deltas]
        sd_delta = round(float(statistics.stdev(delta_vals)), 6) if len(delta_vals) >= 2 else 0.0
        effective_floor = round(max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR), 6)
        delta = round(float(statistics.fmean(delta_vals)), 6) if delta_vals else round(
            float(paired_mean) - float(unpaired_mean), 6)

        verdict = _novelty_verdict(delta, effective_floor)
        supported = verdict == "supported"
        weakened = verdict == "weakened"
        reversed_effect = verdict == "reversed"

        # Leave-one-out robustness diagnostic: drop each single paired seed in turn, recompute
        # BOTH the mean delta and the noise-scaled floor, and re-derive the verdict. Stable iff
        # every fold agrees with the full-sample verdict. This is the exact check 836c's autopsy
        # did by hand (drop the one extreme unpaired-arm seed, mean flips sign), now automatic.
        if len(delta_vals) >= 3:
            folds = []
            for i, (seed_i, _) in enumerate(deltas):
                remaining = [d for j, (_, d) in enumerate(deltas) if j != i]
                fold_sd = float(statistics.stdev(remaining)) if len(remaining) >= 2 else 0.0
                fold_floor = round(max(EFFECT_SIZE_K * fold_sd, EFFECT_SIZE_ABS_FLOOR), 6)
                fold_mean = round(float(statistics.fmean(remaining)), 6)
                fold_verdict = _novelty_verdict(fold_mean, fold_floor)
                folds.append({
                    "dropped_seed": seed_i, "mean_delta": fold_mean,
                    "effective_floor": fold_floor, "verdict": fold_verdict,
                })
            stable = all(f["verdict"] == verdict for f in folds)
            leave_one_out = {
                "applicable": True, "n_folds": len(folds), "full_sample_verdict": verdict,
                "stable": bool(stable), "folds": folds,
            }
        else:
            leave_one_out = {
                "applicable": False,
                "reason": f"fewer than 3 paired seeds ({len(delta_vals)}) -- leave-one-out needs "
                          "at least 2 remaining points to compute a fold SD",
            }

    if not measurable:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "unknown"
    elif supported:
        label = "novelty_gated_consolidation_behavioural_tagging"
        outcome = "PASS"
        evidence_direction = "supports"
    elif weakened:
        label = "retention_invariant_to_novelty_no_tagging"
        outcome = "FAIL"
        evidence_direction = "weakens"
    else:
        label = "reversed_novelty_effect"
        outcome = "FAIL"
        evidence_direction = "mixed"

    criteria_by_arm = {a: ["retained_fraction_measured"] for a in (_arm_id(p) for p in pairings)}
    criteria_nd = arm_criteria_non_degenerate(criteria_by_arm, gate_agg["per_arm_gate"])

    interpretation = {
        "label": label,
        "preconditions": gate_agg["adjudication_preconditions"],
        "preconditions_scope_note": gate_agg["per_arm_gate"]["preconditions_scope_note"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {"name": "paired_retains_more_than_unpaired", "load_bearing": True,
             "passed": bool(supported), "measured": delta, "threshold": effective_floor,
             "note": (
                 f"effective_novelty_margin = max({EFFECT_SIZE_K} * sd_delta={sd_delta}, "
                 f"{EFFECT_SIZE_ABS_FLOOR}); sd_delta is the SD of the per-seed paired "
                 "(paired - unpaired) retained_fraction delta. Replaces V3-EXQ-836c's fixed "
                 "0.15 literal per the project effect-size-gate convention."
             ) if effective_floor is not None else "unmeasurable"},
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
        "novelty_contrast": {
            "paired_retained_fraction_mean": paired_mean,
            "unpaired_retained_fraction_mean": unpaired_mean,
            "delta_paired_minus_unpaired": delta,
            "sd_delta": sd_delta,
            "effective_novelty_margin": effective_floor,
            "effect_size_k": EFFECT_SIZE_K,
            "effect_size_abs_floor": EFFECT_SIZE_ABS_FLOOR,
            "supported": supported,
            "weakened": weakened,
            "reversed": reversed_effect,
        },
        "leave_one_out": leave_one_out,
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
        "supersedes": SUPERSEDES,
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
        "novelty_contrast": result["novelty_contrast"],
        "leave_one_out": result["leave_one_out"],
        "per_arm": result["per_arm"],
        "per_arm_gate": result["per_arm_gate"],
        "arm_results": result["all_cells"],
        "sleep_driver_pattern": "none",
        "dry_run": bool(dry_run),
        "pairings": list(PAIRINGS),
        "fixed_weak_bc_dose": int(FIXED_WEAK_BC_DOSE),
        "fixed_window_steps": int(FIXED_WINDOW_STEPS),
        "offline_ewc_max_coef": OFFLINE_EWC_MAX_COEF,
        "offline_capture_tau": OFFLINE_CAPTURE_TAU,
        "offline_capture_max": OFFLINE_CAPTURE_MAX,
        "offline_fisher_samples": int(cfg.get("offline_fisher_samples", OFFLINE_FISHER_SAMPLES)),
        "effect_size_k": EFFECT_SIZE_K,
        "effect_size_abs_floor": EFFECT_SIZE_ABS_FLOOR,
        "install_took_majority": INSTALL_TOOK_MAJORITY,
        "min_scorable_arms": MIN_SCORABLE_ARMS,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "retention_probe_every": RETENTION_PROBE_EVERY,
        "substrate": "SD-083 offline policy-consolidation window (ree-v3 42ab95f688)",
        "claim_under_test": "MECH-476 (NOVELTY / behavioural-tagging axis of the retrograde-interference falsifier)",
        "redesign_of": SUPERSEDES,
        "redesign_reason": (
            "V3-EXQ-836c's reversed_novelty_effect verdict was driven entirely by one outlier "
            "seed in the unpaired arm (2-15x every sibling) against a fixed 0.15 margin, and did "
            "not survive a leave-one-out check (failure_autopsy_mech476-mech475-cluster_2026-07-29). "
            "This run replaces the fixed margin with a noise-scaled floor, increases n from 6 to "
            "10 seeds per arm, reports a robust statistic (median) alongside the mean, and runs "
            "the leave-one-out check in-line so the verdict is self-auditing."
        ),
        "reference_build": "128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32, "
                           "unconstrained + SD-083 offline EWC anchor (novelty-gated)",
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--pairings", type=str, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="RL refinement budget per cell")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--probe-every", type=int, default=None)
    parser.add_argument("--fisher-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = tuple(args.seeds) if args.seeds else DRY_SEEDS
        pairings = tuple(args.pairings) if args.pairings else PAIRINGS
        on_budget = args.episodes if args.episodes is not None else fan.DRY_RL
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 3
        steps = args.steps if args.steps is not None else 40
        probe_every = args.probe_every if args.probe_every is not None else DRY_RETENTION_PROBE_EVERY
        fisher_samples = args.fisher_samples if args.fisher_samples is not None else DRY_FISHER_SAMPLES
    else:
        seeds = tuple(args.seeds) if args.seeds else SEEDS
        pairings = tuple(args.pairings) if args.pairings else PAIRINGS
        on_budget = args.episodes if args.episodes is not None else int(
            fan.RL_EPISODES * baselines.REF_BUDGET_MULTIPLIER)
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 20
        steps = args.steps if args.steps is not None else 60
        probe_every = args.probe_every if args.probe_every is not None else RETENTION_PROBE_EVERY
        fisher_samples = args.fisher_samples if args.fisher_samples is not None else OFFLINE_FISHER_SAMPLES

    result = run_experiment(seeds, pairings, on_budget=on_budget, eval_eps=eval_eps,
                            steps=steps, probe_every=probe_every, fisher_samples=fisher_samples)

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cfg: Dict[str, Any] = {
        "pairings": list(pairings),
        "seeds": list(seeds),
        "fixed_weak_bc_dose": int(FIXED_WEAK_BC_DOSE),
        "fixed_window_steps": int(FIXED_WINDOW_STEPS),
        "rl_budget_per_cell": int(on_budget),
        "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps),
        "retention_probe_every": int(probe_every),
        "offline_ewc_max_coef": OFFLINE_EWC_MAX_COEF,
        "offline_capture_tau": OFFLINE_CAPTURE_TAU,
        "offline_capture_max": OFFLINE_CAPTURE_MAX,
        "offline_fisher_samples": int(fisher_samples),
        "representation": baselines.REF_REPRESENTATION,
        "reference_budget_multiplier": baselines.REF_BUDGET_MULTIPLIER,
        "effect_size_k": EFFECT_SIZE_K,
        "effect_size_abs_floor": EFFECT_SIZE_ABS_FLOOR,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
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
    nc = result["novelty_contrast"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"evidence={result['evidence_direction']} non_degenerate={result['non_degenerate']} "
        f"scorable_arms={result['n_scorable_arms']}", flush=True,
    )
    for pairing in pairings:
        r = result["per_arm"][_arm_id(pairing)]
        print(
            f"  {r['arm_id']}: pairing={r['offline_novelty_pairing']} "
            f"install_took={r['n_seeds_install_took']}/{r['n_cells']} "
            f"post_bc_mean={r['post_bc_foraging_competence_mean']} "
            f"novelty_factor={r['offline_novelty_factor_mean']} "
            f"measured_novelty={r['offline_measured_novelty_mean']} "
            f"retained_frac_mean={r['retained_fraction_mean_over_took']} "
            f"retained_frac_median={r['retained_fraction_median_over_took']} "
            f"green={r['gate_green']} peak_mean={r['trajectory_peak_mean']}", flush=True,
        )
    print(
        f"  novelty_contrast: paired={nc['paired_retained_fraction_mean']} "
        f"unpaired={nc['unpaired_retained_fraction_mean']} "
        f"delta={nc['delta_paired_minus_unpaired']} sd_delta={nc['sd_delta']} "
        f"effective_novelty_margin={nc['effective_novelty_margin']} "
        f"supported={nc['supported']} weakened={nc['weakened']} reversed={nc['reversed']}",
        flush=True,
    )
    loo = result["leave_one_out"]
    if loo.get("applicable"):
        print(
            f"  leave_one_out: n_folds={loo['n_folds']} stable={loo['stable']} "
            f"full_sample_verdict={loo['full_sample_verdict']}", flush=True,
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
