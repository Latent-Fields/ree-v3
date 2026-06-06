#!/opt/local/bin/python3
"""V3-EXQ-643b -- modulatory-authority validation with a CORRECTED C2 (reuse-preserving).

experiment_purpose=diagnostic. claim_ids=[]. Routed from
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-647_2026-06-06.{md,json}.

WHAT THIS CORRECTS
------------------
V3-EXQ-647 (the arm-reuse reconstruction of 643a on cloud-4) PASSed readiness + C0
+ C1 + C3 but FAILed C2. The autopsy (user-accepted "as stated") found the C2 FAIL
was a TEST-DESIGN / measurement limitation, NOT an authority falsification:

  (1) The 643a C2 "rank-change" metric compares the ON arms' bias_changed_selection_frac
      against the OFF arm (ARM_A) and requires ON - OFF > 0.05 on >= 2/3 seeds. But the
      OFF arm already has a HIGH, wildly seed-variable bias_changed_selection_frac
      (0.073 / 0.331 / 0.608 in 647) because the OFF arm still runs the other bias
      channels / the legacy normalize_score_bias_to_e3_range path. The authority, when
      ON, REPLACES that legacy normalization -- so C2 demanded a gain<1 gap-relative
      authority ADD selection-change on top of a saturated, noisy baseline. Confounded.

  (2) The 643a self-route mapped C2-fail to "sweep gain higher". The within-run C3
      datum REFUTES that: gain 0.5 -> 0.8 raised scale_factor 3.28 -> 18.0 (5.5x) with
      bias_changed_selection_frac FLAT (0.320 -> 0.350). A gain<1 gap-relative authority
      flips only near-ties, whose frequency is fixed by the environment / primary-score
      geometry, NOT by the gain. So higher gain cannot help.

THE CORRECTED C2 (this script)
------------------------------
The confound was always in the ON-OFF SUBTRACTION, never inside the ON arm. Because the
authority fires on ~0.97 of ON-arm ticks (C1), an ON-arm tick with
selected_candidate_rank_before_bias > 0 IS the authority-path moving the committed
selection off the raw-best -- authority-attributable WITHOUT comparing to the OFF arm.
So the corrected C2 is measured WITHIN the ON arms, in ABSOLUTE terms:

  C2a (authority-path carves committed selection, LOAD-BEARING): on each ON arm, on
      >= MIN_SEEDS seeds, the authority is ACTIVE (modulatory_authority_active_frac >
      AUTHORITY_ACTIVE_FRAC_FLOOR) AND it changes the committed selection on a non-trivial
      fraction of ticks (bias_changed_selection_frac > C2A_FLIP_FLOOR). No OFF subtraction.

  C2b (dose-response on flips, INFORMATIVE -- like C3): does the higher-gain arm change
      selection MORE than the lower-gain arm (mean C flip-rate - mean B flip-rate >
      C2B_DOSE_MARGIN)? Expected FLAT per the 647 C3 datum -- this is the quantitative
      restatement of "sweep gain higher does not help".

Overall PASS = readiness AND C0 AND C1 AND C2a. C2b is informative (does not gate PASS),
exactly as 643a's C3 dose-response was informative. The legacy ON-OFF c2_* fields are
retained in the summary for provenance/comparison but DO NOT drive the verdict.

HONEST SCOPE (what C2a does and does NOT establish)
---------------------------------------------------
C2a establishes that the modulatory-authority PATH (the rescaled combined modulatory
contribution: dACC / curiosity / vigor / lateral_pfc / ofc / mech295 / MECH-341, gated
by the authority) carves committed selection at a non-trivial, near-tie-bound rate. It
does NOT isolate whether the gap-relative RESCALING specifically (vs the native
modulatory channels at their unscaled magnitude) is what causes the carving -- that
requires a per-tick argmin(raw+scale*mod) vs argmin(raw+mod) comparison the E3 selector
does not currently emit. Adding that diagnostic changes ree_core (and therefore the arm
fingerprint / substrate_hash), which would break byte-reuse of the 646 baseline. So the
rescaling-vs-native isolation is the DEFERRED "clean-later" pass (read-only selector
diagnostic + a fresh OFF mint on the modified substrate). raw_score_range_mean per arm is
reported as the near-tie indicator (ON arms run with small raw_score_range under SD-056
clamping, i.e. in the near-tie regime the authority is designed to act in).

REUSE (identical to 647; preserves the 646 baseline)
----------------------------------------------------
ARM_A (control, authority OFF) is REUSED by explicit cite from the latest V3-EXQ-646
mint (same as 647) -- ARM_A config is UNCHANGED, so the 646 cells remain byte-reusable
on cloud-4. ARM_B (gain 0.5) / ARM_C (gain 0.8) run fresh via 643a's OWN _run_seed_arm
(byte-identical ON-arm computation -- this script changes ONLY the evaluator, never the
rollout). ARM_A reuse stays load-bearing for the C1 off-silent check.

architecture_epoch: "ree_hybrid_guardrails_v1"

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_643b_modulatory_authority_corrected_c2.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_643b_modulatory_authority_corrected_c2"
QUEUE_ID = "V3-EXQ-643b"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_647_modulatory_authority_reuse_split"  # corrects the shared C2 design

# Corrected-C2 thresholds (pre-registered).
C2A_FLIP_FLOOR = 0.10    # authority must change the committed selection on > 10% of ON-arm ticks
C2B_DOSE_MARGIN = 0.05   # higher-gain arm must change selection >5pp more than lower-gain (informative)

# Lineage source for the reused control arm (same as 647).
CITE_MINT_EXPERIMENT_TYPE = "v3_exq_646_mint_modulatory_authority_off_baseline"
EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# OFF-arm metric keys the evaluator reads off the control cells. A reused control
# cell missing any of these is unusable -> refuse the reuse.
NEEDED_OFF_KEYS: Tuple[str, ...] = (
    "seed",
    "modulatory_authority_active_frac",
    "modulatory_authority_scale_factor_mean",
    "modulatory_authority_range_mean",
    "raw_score_range_mean",
    "raw_bounded_frac",
    "score_bias_abs_mean",
    "bias_changed_selection_frac",
    "visited_cells",
    "mean_episode_length",
    "selected_action_class_entropy",
    "n_positive_control_ticks",
)


def _load_643a_module():
    """Import the 643a module so the ON-arm computation + base evaluator are byte-identical."""
    path = REPO_ROOT / "experiments" / "v3_exq_643a_modulatory_authority_validation.py"
    spec = importlib.util.spec_from_file_location("exq643a_for_corrected_c2", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _latest_mint_manifest() -> Optional[Path]:
    cands = sorted(EVIDENCE_DIR.glob(f"{CITE_MINT_EXPERIMENT_TYPE}_*_v3.json"))
    return cands[-1] if cands else None


def _reused_control_cells(
    m, seeds: List[int], p0: int, p1: int, steps: int, dry_run: bool
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Obtain ARM_A (control) cells by explicit-cite reuse of the latest 646 mint.

    Identical reuse path to V3-EXQ-647. Raises RuntimeError on a real (non-dry) run if
    no valid mint manifest is available -- the runner classifies that as ERROR and
    leaves the item in the queue (correct: do not assemble a 643a without its control).
    """
    manifest_path = _latest_mint_manifest()

    if manifest_path is None:
        if dry_run:
            print(
                "Dry run -- no V3-EXQ-646 mint manifest found; computing control locally "
                "as a stand-in.",
                flush=True,
            )
            from experiments._lib.baselines.exq643_modulatory_authority_baseline import (
                run_off_cell,
            )
            off_cells = []
            for seed in seeds:
                print(f"Seed {seed} Condition ARM_A (dry-run local stand-in)", flush=True)
                cell = run_off_cell(
                    seed, p0_episodes=p0, p1_episodes=p1, steps_per_episode=steps
                )
                cell["reused_from_run_id"] = "DRY_RUN_LOCAL"
                off_cells.append(cell)
                print("verdict: PASS", flush=True)
            return off_cells, {"reuse_source": "dry_run_local_off", "mint_manifest": None}
        raise RuntimeError(
            "No V3-EXQ-646 mint manifest found under evidence/experiments/ -- "
            "the control arm must be minted (646) before this reuse reconstruction can run."
        )

    with open(manifest_path) as fh:
        mint = json.load(fh)
    mint_run_id = mint.get("run_id")
    rows = [
        r for r in (mint.get("arm_results") or [])
        if isinstance(r, dict) and r.get("arm_id") == "ARM_A"
    ]
    by_seed = {int(r["seed"]): r for r in rows if "seed" in r}

    off_cells: List[Dict[str, Any]] = []
    for seed in seeds:
        cell = by_seed.get(int(seed))
        if cell is None:
            raise RuntimeError(
                f"Mint {mint_run_id} has no ARM_A cell for seed {seed}; cannot reuse."
            )
        if cell.get("error_note") is not None:
            raise RuntimeError(
                f"Mint {mint_run_id} ARM_A seed {seed} carried error_note; refusing reuse."
            )
        missing = [k for k in NEEDED_OFF_KEYS if k not in cell]
        if missing:
            raise RuntimeError(
                f"Mint {mint_run_id} ARM_A seed {seed} missing keys {missing}; refusing reuse."
            )
        reused = json.loads(json.dumps(cell, default=str))
        fp = reused.get("arm_fingerprint") or {}
        reused["reused_from_run_id"] = mint_run_id
        reused["reused_fingerprint"] = (
            fp.get("arm_fingerprint") if isinstance(fp, dict) else None
        )
        reused["reused_at_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        off_cells.append(reused)
        print(f"Seed {seed} Condition ARM_A (REUSED from {mint_run_id})", flush=True)
        print("verdict: PASS", flush=True)

    reuse_meta = {
        "reuse_source": "explicit_cite_646_mint",
        "mint_manifest": manifest_path.name,
        "mint_run_id": mint_run_id,
        "mint_machine_classes": mint.get("machine_classes"),
        "n_control_cells_reused": len(off_cells),
    }
    return off_cells, reuse_meta


def _evaluate_corrected_c2(m, arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run 643a's base evaluator (readiness / C0 / C1 / C3 / per_arm_means + legacy c2),
    then layer the corrected, within-ON-arm C2a (load-bearing) + C2b (informative) on top
    and override overall_pass with the corrected criterion. The legacy ON-OFF c2_* keys
    are retained for provenance but DO NOT gate the verdict."""
    base = dict(m._evaluate(arm_results))

    b_rows = m._arm_rows(arm_results, "ARM_B")
    c_rows = m._arm_rows(arm_results, "ARM_C")

    def _carve_seeds(rows: List[Dict[str, Any]]) -> int:
        return sum(
            1 for r in rows
            if float(r.get("modulatory_authority_active_frac", 0.0)) > m.AUTHORITY_ACTIVE_FRAC_FLOOR
            and float(r.get("bias_changed_selection_frac", 0.0)) > C2A_FLIP_FLOOR
        )

    c2a_carve_b = _carve_seeds(b_rows)
    c2a_carve_c = _carve_seeds(c_rows)
    c2a_pass = (c2a_carve_b >= m.MIN_SEEDS) and (c2a_carve_c >= m.MIN_SEEDS)

    b_flip_mean = m._mean_key(b_rows, "bias_changed_selection_frac")
    c_flip_mean = m._mean_key(c_rows, "bias_changed_selection_frac")
    c2b_dose_delta = c_flip_mean - b_flip_mean
    c2b_dose_pass = c2b_dose_delta > C2B_DOSE_MARGIN

    # Near-tie context (NOT a gate): ON arms run with small raw_score_range under SD-056
    # clamping == the near-tie regime the gain<1 gap-relative authority is designed to act in.
    b_range_mean = m._mean_key(b_rows, "raw_score_range_mean")
    c_range_mean = m._mean_key(c_rows, "raw_score_range_mean")

    # Corrected overall PASS: readiness AND C0 AND C1 AND C2a (C2b informative, like C3).
    corrected_overall_pass = bool(
        base["readiness_met"] and base["c0_pass"] and base["c1_pass"] and c2a_pass
    )

    # Non-degeneracy of the corrected load-bearing criterion: meaningful only when the
    # authority actually fired (readiness met) and both ON arms produced rank-valid ticks.
    c2a_non_degenerate = bool(
        base["readiness_met"] and len(b_rows) > 0 and len(c_rows) > 0
    )

    base.update({
        # Corrected C2 (this script's contribution).
        "C2A_FLIP_FLOOR": C2A_FLIP_FLOOR,
        "C2B_DOSE_MARGIN": C2B_DOSE_MARGIN,
        "c2a_carve_seeds_B": c2a_carve_b,
        "c2a_carve_seeds_C": c2a_carve_c,
        "c2a_authority_carves_selection_pass": bool(c2a_pass),
        "c2a_non_degenerate": c2a_non_degenerate,
        "c2b_flip_mean_B": round(b_flip_mean, 6),
        "c2b_flip_mean_C": round(c_flip_mean, 6),
        "c2b_dose_delta_C_minus_B": round(c2b_dose_delta, 6),
        "c2b_dose_response_on_flips_pass": bool(c2b_dose_pass),
        "near_tie_raw_score_range_mean_B": round(b_range_mean, 6),
        "near_tie_raw_score_range_mean_C": round(c_range_mean, 6),
        # Legacy ON-OFF C2 retained for provenance ONLY (does not gate the corrected verdict).
        "legacy_c2_rank_change_pass": bool(base.get("c2_pass", False)),
        "corrected_c2_supersedes_legacy_c2": True,
        # Override the verdict with the corrected criterion.
        "overall_pass": corrected_overall_pass,
        "overall_pass_basis": "readiness AND C0 AND C1 AND C2a (corrected within-ON-arm)",
    })
    return base


def _corrected_label(summary: Dict[str, Any]) -> str:
    if not summary["readiness_met"]:
        return "substrate_not_ready_requeue"
    if summary["overall_pass"] and summary["c2b_dose_response_on_flips_pass"]:
        return "authority_carves_selection_dose_responsive"
    if summary["overall_pass"] and not summary["c2b_dose_response_on_flips_pass"]:
        return "authority_carves_selection_dose_flat_near_tie_bound"
    if summary["c1_pass"] and not summary["c2a_authority_carves_selection_pass"]:
        return "authority_active_but_no_selection_change"
    return "authority_does_not_fire_on_bounded_real_range"


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    m = _load_643a_module()
    seeds = m.DRY_RUN_SEEDS if dry_run else m.SEEDS
    p0 = m.DRY_RUN_P0 if dry_run else m.P0_WARMUP_EPISODES
    p1 = m.DRY_RUN_P1 if dry_run else m.P1_MEASUREMENT_EPISODES
    steps = m.DRY_RUN_STEPS if dry_run else m.STEPS_PER_EPISODE

    # 1. Control arm: REUSED from the 646 mint (explicit cite) -- ARM_A config UNCHANGED.
    off_cells, reuse_meta = _reused_control_cells(m, seeds, p0, p1, steps, dry_run)

    # 2. Experimental arms: fresh, byte-identical to 643a's ON arms (verbatim _run_seed_arm).
    arm_results: List[Dict[str, Any]] = list(off_cells)
    on_arms = [a for a in m.ARMS if a["arm_id"] != "ARM_A"]
    for arm in on_arms:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = m._run_seed_arm(arm, seed, p0, p1, steps)
            arm_results.append(cell)
            passed = cell.get("error_note") is None
            print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)

    # 3. Corrected C2 evaluation (within-ON-arm; legacy ON-OFF C2 retained for provenance).
    summary = _evaluate_corrected_c2(m, arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    label = _corrected_label(summary)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "modulatory_range_supra_floor",
                "description": (
                    "gate's true cross-candidate modulatory range (modulatory_authority_range) "
                    "clears the floor -- the SAME RANGE statistic C1 (and thus the C2a authority "
                    "precondition) gates on; NOT a magnitude proxy"
                ),
                "control": "ON arms, ticks with >= 2 first-action classes",
                "measured": round(summary["p_range_measured"], 8),
                "threshold": m.MODULATORY_RANGE_FLOOR,
                "met": bool(summary["p_range_met"]),
            },
            {
                "name": "primary_scores_bounded",
                "description": (
                    "fraction of ON-arm P1 ticks with e3_raw_score_range_mean < "
                    f"{m.RAW_SCORE_RANGE_BOUND} -- non-vacuity guard"
                ),
                "control": "SD-056 rollout-norm clamp enabled",
                "measured": round(summary["p_bounded_measured"], 6),
                "threshold": m.BOUNDED_FRAC,
                "met": bool(summary["p_bounded_met"]),
            },
            {
                "name": "authority_active_on_positive_control",
                "description": (
                    "authority FIRES on the ON arms (modulatory_authority_active_frac > "
                    "AUTHORITY_ACTIVE_FRAC_FLOOR) -- the readiness precondition for the C2a "
                    "flip-rate criterion, asserting the SAME count/frequency kind of statistic "
                    "C2a routes on (NOT a magnitude). If the authority never fires, the "
                    "ON-arm flip rate is meaningless -> substrate_not_ready_requeue"
                ),
                "control": "ON arms (gain 0.5 / 0.8) on bounded scores",
                "measured": round(
                    min(
                        summary["per_arm_means"]["ARM_B"]["modulatory_authority_active_frac"],
                        summary["per_arm_means"]["ARM_C"]["modulatory_authority_active_frac"],
                    ),
                    6,
                ),
                "threshold": m.AUTHORITY_ACTIVE_FRAC_FLOOR,
                "met": bool(summary["c1_pass"]),
            },
        ],
        "criteria_non_degenerate": {
            "C0": bool(summary["c0_pass"]),
            "C1": bool(summary["c1_non_degenerate"]),
            "C2a": bool(summary["c2a_non_degenerate"]),
        },
        "criteria": [
            {
                "name": "C2a_authority_path_carves_selection",
                "load_bearing": True,
                "passed": bool(
                    summary["c2a_authority_carves_selection_pass"] and summary["readiness_met"]
                ),
            },
        ],
        "grid": {
            "substrate_not_ready_requeue": "readiness fails (modulatory_range below floor OR scores unbounded) -- stabilize, NOT a falsification.",
            "authority_carves_selection_dose_responsive": "C2a PASS + C2b dose-response holds: authority carves committed selection AND more gain carves more. Full validation.",
            "authority_carves_selection_dose_flat_near_tie_bound": "C2a PASS + C2b dose FLAT: authority carves committed selection at a non-trivial but NEAR-TIE-BOUND rate; higher gain does NOT carve more (the env near-tie frequency, not the gain, sets the flip rate). NOT remedied by higher gain. Rescaling-vs-native isolation deferred to the clean-later ree_core-diagnostic + re-mint pass.",
            "authority_active_but_no_selection_change": "C1 PASS but C2a FAIL: the authority fires but never changes the committed choice -- genuine impotence; escalate to the clean-later rescaling-isolation pass.",
            "authority_does_not_fire_on_bounded_real_range": "readiness met but C1 fails -- the gate does not fire on a real bounded range; /diagnose-errors on the rescale arithmetic.",
        },
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "result": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": SUPERSEDES,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Corrected-C2 re-validation of the modulatory-bias-selection-authority substrate "
            "(routed from failure_autopsy_V3-EXQ-647). Reuse-preserving: ARM_A reused by "
            "explicit cite from the V3-EXQ-646 mint (config UNCHANGED -> byte-reusable on "
            "cloud-4); ARM_B/ARM_C fresh via 643a's verbatim _run_seed_arm (ONLY the evaluator "
            "differs). The confounded 643a/647 C2 (ON minus OFF rank-change vs a legacy-channel-"
            "saturated OFF baseline) is REPLACED by a within-ON-arm criterion: C2a (LOAD-BEARING) "
            "= authority active AND bias_changed_selection_frac > C2A_FLIP_FLOOR on >= MIN_SEEDS "
            "seeds per ON arm (authority-attributable because the authority fires on ~0.97 of ON "
            "ticks, so rank_before_bias>0 is the authority moving the committed choice off the "
            "raw-best); C2b (INFORMATIVE) = dose-response on flips (higher gain carves more?). "
            "The 'sweep gain higher' grid branch is dropped (the 647 C3 datum -- 5.5x scale, flat "
            "flip rate -- refutes it). HONEST SCOPE: C2a establishes the modulatory-authority PATH "
            "carves committed selection; it does NOT isolate whether the gap-relative RESCALING "
            "(vs native modulatory channels) causes the carving -- that needs a per-tick "
            "argmin(raw+scale*mod) vs argmin(raw+mod) selector diagnostic which would change "
            "substrate_hash and break 646 reuse, so it is the DEFERRED clean-later pass (read-only "
            "ree_core diagnostic + fresh OFF mint). diagnostic, claim_ids=[]; does NOT weight "
            "governance. A corrected-C2 PASS is the signal governance MAY use to revisit the "
            "modulatory-bias-selection-authority substrate_queue readiness gate -- this run does "
            "not write it. Cross-class verdict comparison: the Mac full V3-EXQ-643a."
        ),
        "interpretation": interpretation,
        "dry_run": bool(dry_run),
        "corrected_c2": {
            "rationale": "failure_autopsy_V3-EXQ-647_2026-06-06 section 7.1",
            "C2A_FLIP_FLOOR": C2A_FLIP_FLOOR,
            "C2B_DOSE_MARGIN": C2B_DOSE_MARGIN,
            "rescaling_vs_native_isolation": "deferred (clean-later ree_core diagnostic + re-mint)",
        },
        "reuse_test": {
            "reused_arm_id": "ARM_A",
            "fresh_arm_ids": [a["arm_id"] for a in on_arms],
            "control_arm_source": "V3-EXQ-646 mint (explicit cite)",
            "reuse_phase": 1,
            "reuse_mode": "explicit_cite",
            **reuse_meta,
            "compare_to_full_fresh": "V3-EXQ-643a (DLAPTOP-4.local, darwin-arm64)",
        },
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": p0,
            "p1_measurement_episodes": p1,
            "steps_per_episode": steps,
            "env_kwargs": m.ENV_KWARGS,
            "arms": [
                {
                    "arm_id": a["arm_id"],
                    "label": a["label"],
                    "use_modulatory_selection_authority": a["use_modulatory_selection_authority"],
                    "modulatory_authority_gain": a["modulatory_authority_gain"],
                    "source": "reused_from_646" if a["arm_id"] == "ARM_A" else "fresh_cloud4",
                }
                for a in m.ARMS
            ],
        },
        "acceptance_criteria": {
            "readiness_met": summary["readiness_met"],
            "C0_curiosity_non_degeneracy": summary["c0_pass"],
            "C1_authority_mechanism_active": summary["c1_pass"],
            "C2a_authority_path_carves_selection": summary["c2a_authority_carves_selection_pass"],
            "C2b_dose_response_on_flips_informative": summary["c2b_dose_response_on_flips_pass"],
            "C3_dose_response_scale_informative": summary["c3_dose_response_pass"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = EVIDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
        print(f"Result written to: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}  label: {label}", flush=True)
    print(f"  reuse_source: {reuse_meta.get('reuse_source')}", flush=True)
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-643b modulatory-authority corrected-C2 re-validation (reuse-preserving)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
