#!/opt/local/bin/python3
"""V3-EXQ-836e -- MECH-476 falsifier (INTERVAL axis, REDESIGN): is competence RETENTION strengthened
by an elapsed OFFLINE A->B INTERVAL -- a genuine consolidation PROCESS -- or only by a concurrent
constraint coefficient? Same question as V3-EXQ-836b; the DISCRIMINATION RULE is corrected, not the
design.

SUPERSEDES V3-EXQ-836b. WHY. failure_autopsy_v3-exq-836b_2026-07-29.md diagnosed 836b's FAIL
(non_monotone_interval_response) as a measurement/test-design defect, not a genuine non-monotone
interval-response: the SUPPORTED/WEAKENED/NON-MONOTONE call rested on a FIXED literal
RESISTANCE_INTERVAL_MARGIN=0.15, not scaled to the observed per-arm noise (per-arm SEMs at n=6 ran
0.098-0.147, comparable in magnitude to the between-arm deltas [+0.147, +0.213, -0.157] the verdict
was read from), and the non-monotone peak at n400 does not survive a leave-one-out check dropping
the single most extreme seed (n400's mean is driven by one seed at retained_fraction=1.285 against
siblings in 0.202-0.934; dropping it pulls the mean 0.749 -> 0.642, collapsing the peak). This is
the SAME defect as siblings V3-EXQ-836 (dose) and V3-EXQ-836c (novelty), both already redone under
this rule (V3-EXQ-836a, V3-EXQ-836d) -- this leg completes the family. Per the project's own
documented effect-size-gate convention (scale noise on the SD of the delta between arms, never a
bare fixed number -- see v3_exq_824a_q081_shared_organisation_landmark_removal.py:EFFECT_SIZE_K/
EFFECT_SIZE_ABS_FLOOR, and the identical redesign already applied in v3_exq_836a /
v3_exq_836d_..._redesign.py), 836b's discrimination rule violated that convention. This redesign
corrects it and re-runs the SAME design at higher power, coordinated with 836a/836d (identical
EFFECT_SIZE_K=1.5 / EFFECT_SIZE_ABS_FLOOR=0.05, identical seed COUNT of 10, identical leave-one-out
+ median-alongside-mean reporting shape) so all three MECH-476 arms share one discrimination
standard.

WHAT CHANGED FROM V3-EXQ-836b (repair pathway, both applied; per the cluster/836b autopsies'
"and/or"):
  1. RESISTANCE_INTERVAL_MARGIN (a fixed 0.15 literal) -> `effective_interval_margin`, a
     noise-scaled floor `max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR)` computed from the SD
     of the per-seed PAIRED delta between the min- and max-interval (N=0 vs N=900) arms (same seed
     integer -> same RNG seed at cell construction in both arms, so a per-seed pairing is available
     exactly as in the original design's shared SEEDS tuple). Mirrors the 824a / 836a / 836d
     convention verbatim.
  2. SEEDS: 6 -> 10 per arm (n>=10, the family floor 836a/836d already adopted).
  3. Robust statistic: `retained_fraction_median_over_took` reported per arm ALONGSIDE the mean
     (never replacing it), so a reader can see whether the mean is outlier-driven -- this is exactly
     what exposed 836b's n400 single-seed artifact in the autopsy.
  4. Leave-one-out robustness diagnostic (`leave_one_out` block): for the min/max-interval paired
     delta, recompute BOTH the mean delta and the noise-scaled floor with each single seed dropped
     in turn, and record whether the three-way verdict (supported/weakened/non_monotone) is STABLE
     across every fold. This is the exact check the autopsy applied by hand to the n400 outlier, now
     run automatically and pre-registered so the verdict is falsifiable rather than asserted.

CLAIM UNDER TEST (unchanged from 836b). MECH-476 (candidate / v3_pending, split_from MECH-457):
acquiring competence and retaining it are dissociable capabilities, and -- its distinctive, testable
content -- consolidation is properly defined as RESISTANCE TO RETROGRADE INTERFERENCE (Krakauer,
Ghez & Ghilardi 2005, J Neurosci 25:473), hence a process that should strengthen with elapsed
offline INTERVAL. V3-EXQ-780 ran a single BC dose STRAIGHT into RL with no interval, so REE has
never (until this family) tested for an offline consolidation process at all -- only for concurrent
regularisation (788 critic, 792 KL anchor).

WHAT THIS LEG TESTS, AND WHAT IT DELIBERATELY DOES NOT (unchanged from 836b). The offline INTERVAL,
holding the install dose (300 BC episodes, the ~20.9 took-point 780 measured) and the interfering
unconstrained RL refinement FIXED. It varies offline_window_steps N in {0, 150, 400, 900} and
measures the retained-fraction TRAJECTORY. N=0 is the EMBEDDED unconstrained control (capture
c(0)=0 -> no anchor -> identical to 780/836/836a's OFF arm). The other two arms of MECH-476's
falsifier (DOSE: 836a; NOVELTY: 836d) are separate legs on the SD-083 offline-consolidation-window
substrate; this leg alone is a complete test of the claim's load-bearing interval-dependence
prediction.

HYPOTHESIS (Krakauer's elapsed-time result, ported; unchanged). Krakauer 2005: consolidation
(resistance to a 5-min counter-rotation) emerges with elapsed time after acquisition. Ported: a
BC-installed policy given a LONGER offline consolidation window should retain a LARGER fraction of
its installed competence through the same unconstrained RL refinement, saturating (the capture
curve is bounded).

DESIGN (unchanged). Four arms, all BC-install the raw_view policy at the SHARED lineage baseline
dose (300 via baselines.mech457_retention: build_off_arm -> install_bc_prior(n_bc=300)), then run
the SD-083 offline window at N steps (baselines.consolidate_offline_window), then run the IDENTICAL
unconstrained reference RL refinement (128-wide / 3x budget / z_world detached / credit-replay 3 /
topk 32, NO KL anchor, NO distributional critic) under the window's offline EWC anchor. The ONLY
thing that varies is the offline interval N:
  * window_n0   -- N=0 offline steps (the unconstrained control; c=0 -> no anchor).
  * window_n150 -- N=150 (below the tau=200 knee; partial capture ~0.53).
  * window_n400 -- N=400 (~2x tau; capture ~0.86).
  * window_n900 -- N=900 (deep saturation; capture ~0.99).
Fisher QUALITY (offline_fisher_samples=256) is held FIXED across arms on purpose, so the interval
effect is the capture clock, NOT a Fisher-noise artifact.

DV -- a TRAJECTORY, never a terminal scalar (unchanged). retention_probe_every wires the
substrate's mid-training competence probe; the DV is retained_fraction = terminal_trajectory_
competence / post_bc_installed_competence (baselines.retained_fraction; None when the install was
~0 so an un-taken install cannot manufacture a maximal-erosion reading).

SCORING (pre-registered, discrimination rule CORRECTED).
  * SUPPORTED (outcome PASS, evidence supports MECH-476): over the arms whose install TOOK,
    retained_fraction_mean is monotone non-decreasing in N AND
    (retained_fraction_mean[max_N] - retained_fraction_mean[min_N]) >= effective_interval_margin,
    where effective_interval_margin = max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR) and
    sd_delta is the SD of the per-seed paired (max_N - min_N) retained_fraction delta. Retention is
    then an elapsed-interval-dependent consolidation PROCESS.
  * WEAKENED (outcome FAIL, evidence weakens MECH-476 -> withdraw into MECH-459/460): the retained
    fraction is INVARIANT to the interval (spread < effective_interval_margin). No offline
    consolidation process; retention tracks only the concurrent constraint -- the intended failure
    mode MECH-476 declares up front.
  * NON-MONOTONE (outcome FAIL, evidence mixed): spread >= effective_interval_margin but not
    monotone -- a real but uninterpretable interval effect; routes to /failure-autopsy rather than a
    clean verdict.

PREREQUISITE (NON-NEGOTIABLE, inherited from the retention portfolio, unchanged). An install that
did not take is UNINFORMATIVE about retention. Each arm carries a per-arm install_took strict-
majority readiness precondition (V3-EXQ-785 per-arm discipline via precondition_gate); a red arm
REDS ONLY ITSELF and cannot vacate a sibling. All arms share the SAME 300-episode dose, so
install_took should be uniform across arms -- but the gate stays per-arm for uniformity with
836b/836a/836d. If FEWER THAN TWO arms take, the interval-response is UNMEASURABLE and the run
self-routes substrate_not_ready_requeue, NOT a WEAKENED verdict.

DV-SYMMETRY (mandatory declaration, per arm; unchanged). The DV is retained_fraction (a competence
ratio). The manipulation is the offline interval N. retained_fraction is NOT invariant under "add
offline window steps": a larger N raises the capture resource c(N), which raises the effective EWC
coefficient, which changes the trace-selective protection applied during RL and hence the erosion
the policy suffers. So the manipulation is not a symmetry of the DV, for every arm. This is a
genuine measurement, not an arithmetic identity. (Contrast the N=0 arm: there c=0 so no anchor is
built -- it is a real unconstrained control, not a symmetry-collapsed one.)

RE-DERIVE BRAKE / GOV-REUSE-1: this is a same-question, corrected-instrument re-run of a run whose
own autopsy scored `epistemic_category: measurement_test_design_defect` (NOT `substrate_ceiling`)
with `recommended_substrate_queue_entry.action: none` -- the re-derive brake does not apply (it
gates repeated ceiling hits, not instrument repair), and the decisive readout (a noise-scaled
interval-response verdict) is NOT recorded anywhere: 836b's own manifest computed it under the
wrong rule and at n=6, so there is nothing to reanalyze post-hoc -- the raw per-seed trajectories
exist in 836b's manifest but the redesign also increases N, which 836b's data cannot supply. Must
re-run.

COORDINATION WITH 836a/836d. This is the third and final leg of the MECH-476 three-arm falsifier's
noise-scaled redesign. All three share EFFECT_SIZE_K=1.5, EFFECT_SIZE_ABS_FLOOR=0.05, n=10 seeds
(same seed tuple 42-51), the median-alongside-mean report, and the leave-one-out diagnostic, so a
reader can compare discrimination power across dose/interval/novelty on one standard. 836a is
currently `claimed` (in flight) and 836d `pending` as of this queueing -- neither has produced a
manifest yet, so there is nothing this leg can reanalyze from them (GOV-REUSE-1 checked; each leg's
decisive readout is leg-specific and not derivable from a sibling's).

EXPERIMENT_PURPOSE = evidence. This directly tests MECH-476's load-bearing interval-dependence
prediction and can move the claim toward supported/weakened. Single claim (MECH-476), so no
evidence_direction_per_claim.

ethics_preflight: all-false / allow. No negative valence, no suffering-like state, no self-model, no
offline replay over harm, no social mind / language, no human/clinical data. V3 pre-ethical
instrumentation only (SENT-0). The SD-083 offline window replays the demonstrator's REAL on-expert
states to estimate a Fisher diagonal -- no simulated/hypothesis content is written to any memory
store (MECH-094 N/A).

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF). The SD-083
offline window is a POLICY-consolidation mechanism in the mech457 testbed, distinct from the
cognifold SD-017 sleep loop (which consolidates latents/world-model/self-model, never a policy).
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

EXPERIMENT_TYPE = "v3_exq_836e_mech476_interval_dependent_consolidation_redesign"
QUEUE_ID = "V3-EXQ-836e"
SUPERSEDES = "V3-EXQ-836b"
CLAIM_IDS: List[str] = ["MECH-476"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Offline-interval sweep (the ONLY manipulation) ----------------------------------------
# 0 = unconstrained control (c=0, no anchor); 150 below the tau knee; 400 ~2x tau; 900 deep sat.
WINDOW_STEPS: Tuple[int, ...] = (0, 150, 400, 900)
DRY_WINDOW_STEPS: Tuple[int, ...] = (0, 20, 60)

# Held FIXED across every arm (the install and the offline window's other knobs). Unchanged from
# 836b -- the redesign corrects the discrimination rule, not the manipulation or the substrate.
FIXED_BC_DOSE = 300                    # the ~20.9 install point 780 measured (the lineage OFF dose)
OFFLINE_EWC_MAX_COEF = 100.0           # pre-registered; scaled by capture(N) x novelty(=1.0 here)
OFFLINE_CAPTURE_TAU = 200.0            # capture time-constant (so N=200 ~ 63% capture)
OFFLINE_CAPTURE_MAX = 1.0
OFFLINE_FISHER_SAMPLES = 256           # FIXED across arms so the interval acts through capture, not Fisher noise
DRY_FISHER_SAMPLES = 8

# REDESIGN: n=6 -> n=10 (family floor, matching 836a/836d). Same seed integers reused across arms
# so a per-seed PAIRED delta is available for the noise-scaled floor (matched RNG seed at cell
# construction; the interval itself is what differs downstream).
SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
DRY_SEEDS: Tuple[int, ...] = (42, 43)

RETENTION_PROBE_EVERY = 250
DRY_RETENTION_PROBE_EVERY = 2          # fan.DRY_RL == 6 -> 3 readings under --dry-run

# Pre-registered thresholds (constants, NOT derived from the run's own statistics). The FLOOR
# itself is now noise-scaled (effective_interval_margin, computed in run_experiment); K and
# ABS_FLOOR are the pre-registered scaling constants, IDENTICAL to the sibling redesigns 836a/836d
# so all three MECH-476 arms share one discrimination standard (see COORDINATION note above).
EFFECT_SIZE_K = 1.5
EFFECT_SIZE_ABS_FLOOR = 0.05
INSTALL_TOOK_MAJORITY = 0.5            # >50% of an arm's seeds must have install_took for scorable
MIN_SCORABLE_ARMS = 2                  # fewer than 2 took -> interval-response unmeasurable


def _arm_id(n: int) -> str:
    return f"window_n{int(n)}"


# --- Per-arm readiness precondition (install must TAKE, or the arm is uninformative) --------
PRECONDITION_SPECS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="install_took_strict_majority",
        description=(
            "A strict majority of this interval arm's seeds installed above the competence floor "
            "(post_bc >= COMPETENCE_RESOURCE_FLOOR). An install that did not take is UNINFORMATIVE "
            "about retention; below this the arm reds ONLY ITSELF and never vacates a sibling arm."
        ),
        control="fraction of seeds with install_took=True in this arm, vs 0.5",
        threshold=float(INSTALL_TOOK_MAJORITY),
        direction="lower",   # a FLOOR: met when measured > threshold
        kind="readiness",
    ),
)


def _arm_contexts(steps: Tuple[int, ...]) -> List[Dict[str, Any]]:
    return [{"id": _arm_id(n), "window_steps": int(n)} for n in steps]


def _run_cell(n_steps: int, seed: int, env_kwargs: Dict[str, Any], *,
              on_budget: int, eval_eps: int, steps: int, probe_every: int,
              fisher_samples: int) -> Dict[str, Any]:
    """One (interval x seed) cell: BC-install at the FIXED dose, run the SD-083 offline window at
    N steps, then the SAME unconstrained RL refinement under the window's EWC anchor."""
    arm_id = _arm_id(n_steps)
    print(f"Seed {seed} Condition {arm_id}", flush=True)   # boundary reset for the runner

    # The unconstrained reference RL refinement config, carrying the offline-window knobs so the
    # config_slice (and thus the arm fingerprint) declares the interval.
    cfg = baselines.reference_config(
        on_budget, retention_probe_every=int(probe_every),
        use_offline_consolidation=True,
        offline_window_steps=int(n_steps),
        offline_capture_tau=OFFLINE_CAPTURE_TAU,
        offline_capture_max=OFFLINE_CAPTURE_MAX,
        offline_ewc_max_coef=OFFLINE_EWC_MAX_COEF,
        offline_fisher_samples=int(fisher_samples),
        offline_novelty_pairing="none",     # INTERVAL arm: novelty is not the manipulation
    )

    # Leg-specific config_slice (driver folded into the hash -- not minted for cross-driver reuse,
    # so a false HIT is impossible). The interval is part of the computation, so it is declared.
    slice_cfg: Dict[str, Any] = {
        "arm_id": arm_id,
        "rung_id": fan.RUNG_ID,
        "kind": "mech476_interval_consolidation_redesign",
        "bc_install_dose": int(FIXED_BC_DOSE),
        "offline_window_steps": int(n_steps),
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

        # Phase 1 -- BC install at the FIXED dose, then measure whether it TOOK.
        install = baselines.install_bc_prior(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id,
            n_bc=int(FIXED_BC_DOSE),
        )
        post_bc = float(install["post_bc_foraging_competence"])

        # Phase 1.5 -- the SD-083 OFFLINE CONSOLIDATION WINDOW (N steps). Builds the EWC anchor
        # WITHOUT retraining the policy (post_bc unchanged). N=0 -> anchor None -> unconstrained.
        window = baselines.consolidate_offline_window(
            rep_agent, seed, env_kwargs, steps=steps, cfg=cfg, demo=install["demo"],
        )
        anchor = window.get("anchor")

        # Phase 2 -- the SAME unconstrained RL refinement, under the offline EWC anchor, probed.
        probe_fn = baselines.make_probe_fn(
            rep_agent, seed, env_kwargs, steps=steps, eval_eps=eval_eps, arm_label=arm_id
        )
        guard = baselines.train_off_arm(
            rep_agent, seed, env_kwargs, steps=steps, arm_label=arm_id, cfg=cfg,
            demo=install["demo"], probe_fn=probe_fn, offline_ewc_anchor=anchor,
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
        row["offline_window_steps"] = int(n_steps)
        row["bc_install_dose"] = int(install.get("bc_install_dose", FIXED_BC_DOSE))
        row["post_bc_foraging_competence"] = round(post_bc, 6)
        row["install_took"] = bool(install["install_took"])
        row["bc_warmstart_action_match_recent"] = float(install["bc_warmstart_action_match_recent"])
        # SD-083 window diagnostics (recorded generously -- the mechanism's realised state).
        row["offline_capture_resource"] = float(window.get("capture_resource", 0.0))
        row["offline_novelty_factor"] = float(window.get("novelty_factor", 0.0))
        row["offline_effective_ewc_coef"] = float(window.get("effective_ewc_coef", 0.0))
        row["offline_fisher_mass"] = float(window.get("fisher_mass", 0.0))
        row["offline_n_fisher_states"] = int(window.get("n_fisher_states", 0))
        row["offline_ewc_installed"] = bool(guard.get("offline_ewc_installed", False))
        row["mean_offline_ewc_penalty_recent"] = float(guard.get("mean_offline_ewc_penalty_recent", 0.0))
        row["competence_trajectory"] = trajectory        # FULL per-seed trajectory, never collapsed
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
        f"(arm={arm_id} seed={seed} N={n_steps} post_bc={post_bc} "
        f"capture={row['offline_capture_resource']:.3f} ewc_coef={row['offline_effective_ewc_coef']:.3f} "
        f"retained_fraction={retained} forage/ep={forage})", flush=True,
    )
    return row


def _paired_deltas(min_cells: List[Dict[str, Any]], max_cells: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """Per-seed (max_N - min_N) retained_fraction delta, matched by seed integer. Only seeds where
    BOTH arms installed (install_took) and have a defined retained_fraction contribute -- an
    un-taken install cannot manufacture a delta reading."""
    by_seed_min = {int(c["seed"]): c for c in min_cells
                   if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None}
    by_seed_max = {int(c["seed"]): c for c in max_cells
                   if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None}
    shared = sorted(set(by_seed_min) & set(by_seed_max))
    return [(s, float(by_seed_max[s]["retained_fraction"]) - float(by_seed_min[s]["retained_fraction"]))
            for s in shared]


def _interval_response_verdict(mean_delta: float, effective_floor: float, spread: float,
                                monotone: Optional[bool]) -> str:
    """The pre-registered three-way discrimination, factored so the leave-one-out diagnostic can
    call it identically on each fold. Returns one of 'supported' / 'weakened' / 'non_monotone'."""
    grows = bool(mean_delta >= effective_floor)
    if bool(monotone) and grows:
        return "supported"
    if spread < effective_floor:
        return "weakened"
    return "non_monotone"


def run_experiment(seeds: Tuple[int, ...], window_steps: Tuple[int, ...], *,
                   on_budget: int, eval_eps: int, steps: int, probe_every: int,
                   fisher_samples: int) -> Dict[str, Any]:
    print(
        f"MECH-476 INTERVAL-dependent consolidation falsifier REDESIGN "
        f"({len(window_steps)} interval arms x {len(seeds)} seeds; N={list(window_steps)}; "
        f"rep={baselines.REF_REPRESENTATION}, dose={FIXED_BC_DOSE}, RL_budget={on_budget}, "
        f"probe_every={probe_every}, eval={eval_eps}, steps={steps}; ewc_max_coef={OFFLINE_EWC_MAX_COEF}, "
        f"capture_tau={OFFLINE_CAPTURE_TAU}, fisher_samples={fisher_samples}; "
        f"manipulation=offline INTERVAL ONLY, refinement=unconstrained reference; "
        f"discrimination_margin=noise-scaled max({EFFECT_SIZE_K}*sd_delta, {EFFECT_SIZE_ABS_FLOOR}))",
        flush=True,
    )
    arm_ctxs = _arm_contexts(window_steps)
    # Design-audit BEFORE any compute: the readiness precondition carries no structural bound, so
    # this refuses nothing here -- same guard 792a/785/836/836a/836d use, kept for uniformity.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, arm_ctxs)

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    all_cells: List[Dict[str, Any]] = []
    for n_steps in window_steps:
        for seed in seeds:
            all_cells.append(_run_cell(n_steps, seed, env_kwargs, on_budget=on_budget,
                                       eval_eps=eval_eps, steps=steps, probe_every=probe_every,
                                       fisher_samples=fisher_samples))

    # ---- per-arm retention readouts (trajectory-shaped) -----------------------------------
    def _cells_for(n: int) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == _arm_id(n)]

    per_arm: Dict[str, Any] = {}
    arm_gates: List[Dict[str, Any]] = []
    for n_steps in window_steps:
        arm_id = _arm_id(n_steps)
        cells = _cells_for(n_steps)
        n_cells = len(cells)
        n_install_took = int(sum(1 for c in cells if bool(c.get("install_took", False))))
        install_took_frac = float(n_install_took / n_cells) if n_cells else 0.0
        retained_took = [
            float(c["retained_fraction"]) for c in cells
            if bool(c.get("install_took", False)) and c.get("retained_fraction") is not None
        ]
        retained_mean = round(sum(retained_took) / len(retained_took), 6) if retained_took else None
        # REDESIGN: robust statistic reported ALONGSIDE the mean (never replacing it) -- exactly
        # what exposed 836b's n400 single-seed artifact in the autopsy.
        retained_median = round(statistics.median(retained_took), 6) if retained_took else None
        post_bc_vals = [float(c.get("post_bc_foraging_competence", 0.0)) for c in cells]
        peaks = [float(c.get("trajectory_peak_competence", 0.0)) for c in cells]
        terminals = [float(c.get("trajectory_terminal_competence", 0.0)) for c in cells]
        capture_vals = [float(c.get("offline_capture_resource", 0.0)) for c in cells]
        ewc_coef_vals = [float(c.get("offline_effective_ewc_coef", 0.0)) for c in cells]

        measured = {"install_took_strict_majority": install_took_frac}
        gate = evaluate_arm_gate(arm_id, {"id": arm_id, "window_steps": int(n_steps)},
                                 PRECONDITION_SPECS, measured=measured)
        arm_gates.append(gate)

        per_arm[arm_id] = {
            "arm_id": arm_id,
            "offline_window_steps": int(n_steps),
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
            "offline_capture_resource_mean": round(sum(capture_vals) / n_cells, 6) if n_cells else 0.0,
            "offline_effective_ewc_coef_mean": round(sum(ewc_coef_vals) / n_cells, 6) if n_cells else 0.0,
            "gate_green": bool(gate["gate_green"]),
        }

    gate_agg = aggregate_arm_gates(arm_gates)
    green_arms = list(gate_agg["per_arm_gate"]["green_arms"])       # arms whose install took
    n_scorable = len(green_arms)

    # ---- interval-response scoring over the SCORABLE (install-took) arms -------------------
    scorable_by_n = sorted(
        [(int(per_arm[a]["offline_window_steps"]), per_arm[a]["retained_fraction_mean_over_took"])
         for a in green_arms
         if per_arm[a]["retained_fraction_mean_over_took"] is not None],
        key=lambda t: t[0],
    )
    n_axis = [n for n, _ in scorable_by_n]
    retained_axis = [m for _, m in scorable_by_n]

    measurable = len(scorable_by_n) >= MIN_SCORABLE_ARMS
    non_degenerate = bool(measurable)
    degeneracy_reason = "" if measurable else (
        f"fewer than {MIN_SCORABLE_ARMS} interval arms achieved majority install_took with a "
        f"defined retained_fraction ({len(scorable_by_n)} scorable) -> interval-response "
        f"UNMEASURABLE; substrate_not_ready_requeue, NOT a retention verdict"
    )

    spread = None
    monotone = None
    supported = weakened = non_monotone = False
    effective_floor = None
    sd_delta = None
    mean_delta = None
    leave_one_out: Dict[str, Any] = {"applicable": False}
    if measurable:
        spread = round(max(retained_axis) - min(retained_axis), 6)
        monotone = all(retained_axis[i] <= retained_axis[i + 1] + 1e-9
                       for i in range(len(retained_axis) - 1))

        # REDESIGN: noise-scaled floor, computed from the per-seed PAIRED delta between the
        # min- and max-scorable-interval arms (the comparison the load-bearing criterion reads).
        min_n, max_n = n_axis[0], n_axis[-1]
        deltas = _paired_deltas(_cells_for(min_n), _cells_for(max_n))
        delta_vals = [d for _, d in deltas]
        sd_delta = round(float(statistics.stdev(delta_vals)), 6) if len(delta_vals) >= 2 else 0.0
        effective_floor = round(max(EFFECT_SIZE_K * sd_delta, EFFECT_SIZE_ABS_FLOOR), 6)
        mean_delta = round(float(statistics.fmean(delta_vals)), 6) if delta_vals else round(
            retained_axis[-1] - retained_axis[0], 6)

        verdict = _interval_response_verdict(mean_delta, effective_floor, spread, monotone)
        supported = verdict == "supported"
        weakened = verdict == "weakened"
        non_monotone = verdict == "non_monotone"

        # Leave-one-out robustness diagnostic: drop each single paired seed in turn, recompute
        # BOTH the mean delta and the noise-scaled floor (sd_delta shrinks/grows with the drop),
        # and re-derive the verdict. Stable iff every fold agrees with the full-sample verdict.
        if len(delta_vals) >= 3:
            folds = []
            for i, (seed_i, _) in enumerate(deltas):
                remaining = [d for j, (_, d) in enumerate(deltas) if j != i]
                fold_sd = float(statistics.stdev(remaining)) if len(remaining) >= 2 else 0.0
                fold_floor = round(max(EFFECT_SIZE_K * fold_sd, EFFECT_SIZE_ABS_FLOOR), 6)
                fold_mean = round(float(statistics.fmean(remaining)), 6)
                fold_verdict = _interval_response_verdict(fold_mean, fold_floor, spread, monotone)
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
        label = "consolidation_process_interval_dependent"
        outcome = "PASS"
        evidence_direction = "supports"
    elif weakened:
        label = "retention_invariant_to_interval_no_process"
        outcome = "FAIL"
        evidence_direction = "weakens"
    else:
        label = "non_monotone_interval_response"
        outcome = "FAIL"
        evidence_direction = "mixed"

    criteria_by_arm = {a: ["retained_fraction_measured"] for a in (_arm_id(n) for n in window_steps)}
    criteria_nd = arm_criteria_non_degenerate(criteria_by_arm, gate_agg["per_arm_gate"])

    interpretation = {
        "label": label,
        "preconditions": gate_agg["adjudication_preconditions"],
        "preconditions_scope_note": gate_agg["per_arm_gate"]["preconditions_scope_note"],
        "criteria_non_degenerate": criteria_nd,
        "criteria": [
            {"name": "resistance_grows_with_interval", "load_bearing": True,
             "passed": bool(supported), "measured": mean_delta, "threshold": effective_floor,
             "note": (
                 f"effective_interval_margin = max({EFFECT_SIZE_K} * sd_delta={sd_delta}, "
                 f"{EFFECT_SIZE_ABS_FLOOR}); sd_delta is the SD of the per-seed paired "
                 "(max_N - min_N) retained_fraction delta. Replaces V3-EXQ-836b's fixed 0.15 "
                 "literal per the project effect-size-gate convention."
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
        "interval_response": {
            "window_steps_axis": n_axis,
            "retained_fraction_mean_axis": retained_axis,
            "spread": spread,
            "monotone_non_decreasing": monotone,
            "sd_delta": sd_delta,
            "effective_interval_margin": effective_floor,
            "mean_paired_delta": mean_delta,
            "effect_size_k": EFFECT_SIZE_K,
            "effect_size_abs_floor": EFFECT_SIZE_ABS_FLOOR,
            "supported": supported,
            "weakened": weakened,
            "non_monotone": non_monotone,
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
        "interval_response": result["interval_response"],
        "leave_one_out": result["leave_one_out"],
        "per_arm": result["per_arm"],
        "per_arm_gate": result["per_arm_gate"],
        "arm_results": result["all_cells"],
        "sleep_driver_pattern": "none",
        "dry_run": bool(dry_run),
        # pre-registered constants + provenance
        "window_steps": list(WINDOW_STEPS),
        "fixed_bc_dose": int(FIXED_BC_DOSE),
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
        "claim_under_test": "MECH-476 (INTERVAL axis of the A->B->A retrograde-interference falsifier)",
        "redesign_of": SUPERSEDES,
        "redesign_reason": (
            "V3-EXQ-836b's non_monotone_interval_response verdict rested on a fixed 0.15 margin "
            "not scaled to observed per-arm noise (per-arm SEMs 0.098-0.147 vs deltas "
            "[+0.147,+0.213,-0.157]) and did not survive a leave-one-out check on the n400 arm's "
            "single extreme seed (failure_autopsy_v3-exq-836b_2026-07-29). This run replaces the "
            "fixed margin with a noise-scaled floor, increases n from 6 to 10 seeds per arm, "
            "reports a robust statistic (median) alongside the mean, and runs the leave-one-out "
            "check in-line so the verdict is self-auditing -- identical rule to the sibling "
            "redesigns V3-EXQ-836a (dose) and V3-EXQ-836d (novelty)."
        ),
        "reference_build": "128-wide / 3x budget / z_world detached / credit-replay 3 / topk 32, "
                           "unconstrained (no KL anchor, no distributional critic) + SD-083 offline EWC anchor",
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--window-steps", type=int, nargs="*", default=None)
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
        window_steps = tuple(args.window_steps) if args.window_steps else DRY_WINDOW_STEPS
        on_budget = args.episodes if args.episodes is not None else fan.DRY_RL
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 3
        steps = args.steps if args.steps is not None else 40
        probe_every = args.probe_every if args.probe_every is not None else DRY_RETENTION_PROBE_EVERY
        fisher_samples = args.fisher_samples if args.fisher_samples is not None else DRY_FISHER_SAMPLES
    else:
        seeds = tuple(args.seeds) if args.seeds else SEEDS
        window_steps = tuple(args.window_steps) if args.window_steps else WINDOW_STEPS
        on_budget = args.episodes if args.episodes is not None else int(
            fan.RL_EPISODES * baselines.REF_BUDGET_MULTIPLIER)   # 1000 * 3 = 3000
        eval_eps = args.eval_episodes if args.eval_episodes is not None else 20
        steps = args.steps if args.steps is not None else 60
        probe_every = args.probe_every if args.probe_every is not None else RETENTION_PROBE_EVERY
        fisher_samples = args.fisher_samples if args.fisher_samples is not None else OFFLINE_FISHER_SAMPLES

    result = run_experiment(seeds, window_steps, on_budget=on_budget, eval_eps=eval_eps,
                            steps=steps, probe_every=probe_every, fisher_samples=fisher_samples)

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cfg: Dict[str, Any] = {
        "window_steps": list(window_steps),
        "seeds": list(seeds),
        "fixed_bc_dose": int(FIXED_BC_DOSE),
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
    ir = result["interval_response"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"evidence={result['evidence_direction']} non_degenerate={result['non_degenerate']} "
        f"scorable_arms={result['n_scorable_arms']}", flush=True,
    )
    for n_steps in window_steps:
        r = result["per_arm"][_arm_id(n_steps)]
        print(
            f"  {r['arm_id']}: N={r['offline_window_steps']} "
            f"install_took={r['n_seeds_install_took']}/{r['n_cells']} "
            f"post_bc_mean={r['post_bc_foraging_competence_mean']} "
            f"capture={r['offline_capture_resource_mean']} ewc_coef={r['offline_effective_ewc_coef_mean']} "
            f"retained_frac_mean={r['retained_fraction_mean_over_took']} "
            f"retained_frac_median={r['retained_fraction_median_over_took']} "
            f"green={r['gate_green']} peak_mean={r['trajectory_peak_mean']}", flush=True,
        )
    print(
        f"  interval_response: N={ir['window_steps_axis']} retained={ir['retained_fraction_mean_axis']} "
        f"spread={ir['spread']} monotone={ir['monotone_non_decreasing']} "
        f"sd_delta={ir['sd_delta']} effective_interval_margin={ir['effective_interval_margin']} "
        f"mean_paired_delta={ir['mean_paired_delta']} "
        f"supported={ir['supported']} weakened={ir['weakened']} non_monotone={ir['non_monotone']}",
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
