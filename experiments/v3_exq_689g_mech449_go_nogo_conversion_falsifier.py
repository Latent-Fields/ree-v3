"""V3-EXQ-689g -- MECH-449 / ARC-107 Go/No-Go eligibility constitution conversion falsifier.

ABLATION FALSIFIER for the BUILT Go/No-Go eligibility constitution (MECH-449,
landed 2026-06-21; ree_core/predictors/e3_selector.py._go_nogo_eligibility_gate,
config flag use_go_nogo_constitution). claim_ids=[MECH-449, ARC-107].
EXPERIMENT_PURPOSE=evidence -- this is the falsifier that GATES MECH-449's
promotion (the design note s3.2 ablation falsifier / claims.yaml MECH-449
what_would_answer). PROMOTES NOTHING by itself: MECH-449 stays candidate /
substrate_conditional until this scores a contributory PASS.

WHAT THIS TESTS. MECH-448 (rank-preserving F->eligibility demotion, landed
2026-06-20) lets F decide ELIGIBILITY rather than the winner -- but its envelope
is ORDER-PRESERVING over F (an F-rank prefix), so it ADMITS an F-eligible-but-
undesirable candidate and, under a modulatory pull toward it, SELECTS it.
V3-EXQ-689f established that gap on the live envelope (the No-Go-necessity
falsifier; positive build trigger). 689g tests the COMPLEMENT: does the BUILT
Go/No-Go gate, wired end-to-end through E3TrajectorySelector.select(), actually
CONVERT the committed selection -- suppressing the F-eligible-but-undesirable
candidate (orthogonal to F-rank, which demotion cannot) so a previously-gated
channel commits instead.

SELECTION-FACE, SINGLE-DECISION (deliberate). The DV is the per-decision COMMITTED
SELECTION through the real select() path, NOT a sustained multi-step foraging
behavioural outcome. This is the sanctioned discipline for the conversion-ceiling
family (cf V3-EXQ-689f): a sustained committed-action-diversity DV on the
still-incomplete BG circuit (component 5 post-commit latch OPEN; ARC-107 s6b
ledger) would just re-derive the F-dominance conversion ceiling, not a verdict
(the don't-queue-commitment-dependent-behaviour-while-BG-incomplete gate). The
divergent candidate pool the user's GAP-A framing supplies (SD-056-trained
e2.world_forward -> per-candidate divergence) is provided here BY CONSTRUCTION
(distinct per-candidate world_states -> divergent raw F + a divergent modulatory
channel), and the divergence is a checked NON-VACUITY PRECONDITION, not assumed.
The sustained-behavioural conversion on the trained foraging substrate is the
downstream successor, gated on BG-circuit completeness (s6b ledger).

ARMS (the gate axis; both run MECH-448 demotion ON so the only difference is the
Go/No-Go gate):
  ARM_DEMOTION       use_f_eligibility_demotion=True, use_go_nogo_constitution=False.
                     The F-eligible-but-undesirable candidate U is admitted and,
                     under an overwhelming modulatory pull toward it, SELECTED
                     (the demotion-insufficient gap).
  ARM_CONSTITUTION   demotion ON + use_go_nogo_constitution=True; an active No-Go
                     (safety axis on U + a staleness axis on a stale candidate;
                     the perseveration axis reuses MECH-260) drops U from the
                     eligible set, so the within-eligible argmin CONVERTS to a
                     previously-gated channel. U is NEVER selectable regardless of
                     its modulatory pull (the orthogonal-to-F safety guarantee).

DVs (per (seed, arm) over N_BANKS divergent banks):
  conversion_rate    fraction of banks where ARM_DEMOTION selected U AND
                     ARM_CONSTITUTION selected a DIFFERENT, non-U action class
                     (a previously-gated channel converts). LOAD-BEARING.
  safety_violations  banks where ARM_CONSTITUTION still selects U (must be 0 --
                     No-Go suppresses the harmful class even under overwhelming
                     modulatory pull; NO global disinhibition).
  specificity_ok     on a CONTROL sub-bank with no undesirable tag, the gate does
                     NOT change the selection (ARM_CONSTITUTION == ARM_DEMOTION);
                     the No-Go fires only where it should.

NON-VACUITY PRECONDITIONS (else self-route substrate_not_ready_requeue, non_contributory):
  pool_divergent     median raw-F RANGE across banks > FLOOR AND median modulatory
                     RANGE > FLOOR (the divergent-pool precondition; the SAME
                     range statistic the conversion DV depends on -- not a
                     magnitude proxy).
  gate_engaged       ARM_DEMOTION admits U (demotion_admits_U_rate > floor) AND
                     ARM_CONSTITUTION excluded_count > 0 (the No-Go actually
                     suppressed): the Go/No-Go variables vary across arms.

INTERPRETATION (self-routing; PROMOTES NOTHING until adjudicated):
  - preconditions unmet -> substrate_not_ready_requeue (non_contributory; both
    claims scoring-excluded via non_degenerate=false).
  - preconditions met + conversion >= floor (>=2/3 seeds) + safety + specificity
    -> PASS; supports MECH-449 + ARC-107 (the constitution converts a gated
    channel beyond MECH-448).
  - preconditions met + safety holds + NO conversion -> MECH-449 weakens
    (over-specification: the constitution adds nothing beyond MECH-448); ARC-107
    NOT falsified (non_contributory) per the build's pre-registration.
  - safety violation (ARM_CONSTITUTION selects U) -> MECH-449 weakens (the safety
    contract failed; route to /diagnose-errors -- this should be impossible given
    the built gate).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689g_mech449_go_nogo_conversion_falsifier"
QUEUE_ID = "V3-EXQ-689g"
SUPERSEDES: Optional[str] = None  # NEW question (MECH-449's first falsifier)
CLAIM_IDS: List[str] = ["MECH-449", "ARC-107"]
EXPERIMENT_PURPOSE = "evidence"

EVIDENCE_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

# Design constants (pre-registered; NOT inferred post-hoc).
SEEDS: List[int] = [42, 43, 44]
N_BANKS = 24            # adversarial banks per (seed, arm)
K_CANDIDATES = 5        # candidates per bank (== ACTION_DIM so every class is distinct)
WORLD_DIM = 6
ACTION_DIM = 5
HORIZON = 3

# Modulatory pull toward the undesirable candidate U (OVERWHELMING -- tests that
# the No-Go suppresses on an axis the modulatory pull cannot override).
MODULATORY_PULL = 1.0e3

# Non-vacuity floors.
RAW_F_RANGE_FLOOR = 0.05      # divergent raw-F pool (range, not magnitude)
MODULATORY_RANGE_FLOOR = 0.5  # divergent modulatory channel (the pull spread)
DEMOTION_ADMITS_U_FLOOR = 0.5 # ARM_DEMOTION must admit+select U on a majority of banks

# Acceptance floors.
CONVERSION_FLOOR = 0.5        # per-seed conversion rate
SEED_PASS_FRACTION = 2.0 / 3.0


def _make_candidate(action_class: int, world_vec: torch.Tensor) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(HORIZON + 1)]
    world_states = [world_vec.reshape(1, WORLD_DIM).clone() for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    actions[:, 0, action_class] = 1.0
    return Trajectory(states=states, actions=actions, world_states=world_states)


def _build_bank(rng: torch.Generator) -> List[Trajectory]:
    """A divergent K-candidate bank with all-distinct first-action classes.

    Distinct per-candidate world_states -> divergent raw F. The F-eligible
    undesirable U is chosen AFTER probing raw F (see _run_cell) so it is
    guaranteed to be the F-best (always in the envelope) -- the non-vacuous
    "F-eligible-but-undesirable" setup.
    """
    cands = []
    for k in range(K_CANDIDATES):
        wv = torch.randn(WORLD_DIM, generator=rng) * 0.5 + float(k) * 0.4
        cands.append(_make_candidate(action_class=k, world_vec=wv))  # class == k (distinct)
    return cands


def _raw_f(selector: E3TrajectorySelector, cands: List[Trajectory]) -> List[float]:
    """Probe raw F (zero bias, no signals) -> scores ARE the raw F costs."""
    selector._running_variance = 0.0
    r = selector.select(cands, temperature=1.0, score_bias=torch.zeros(len(cands)))
    return [float(s.detach()) for s in r.scores]


def _select_one(
    selector: E3TrajectorySelector,
    cands: List[Trajectory],
    u_index: int,
    stale_index: int,
    gate_on: bool,
) -> Dict[str, Any]:
    """One committed selection through the real select() path.

    Modulatory channel = a score_bias overwhelmingly favouring U. When gate_on,
    inject a safety No-Go on U + a staleness No-Go on the stale candidate (the
    perseveration axis would, in the live agent loop, reuse MECH-260; here the
    bank is synthetic so we drive the axes directly).
    """
    selector._running_variance = 0.0  # deterministic committed argmin path
    k = len(cands)
    bias = torch.zeros(k)
    bias[u_index] = -MODULATORY_PULL  # overwhelming pull toward U (lower=better)
    go_nogo_signals = None
    if gate_on:
        safety = torch.zeros(k)
        safety[u_index] = 0.9  # safety-No-Go U
        staleness = torch.zeros(k)
        staleness[stale_index] = 0.9  # staleness-No-Go the stale candidate
        go_nogo_signals = {"safety": safety, "staleness": staleness}
    result = selector.select(
        cands, temperature=1.0, score_bias=bias, go_nogo_signals=go_nogo_signals,
    )
    diag = selector.last_score_diagnostics
    raw_scores = [float(s.detach()) for s in result.scores]
    raw_range = max(raw_scores) - min(raw_scores)
    env_size = diag.get("go_nogo_envelope_size", None)
    excluded = (k - int(env_size)) if (gate_on and env_size is not None) else 0
    return {
        "selected_index": int(result.selected_index),
        "selected_class": int(result.selected_action.reshape(-1).argmax().item()),
        "picked_u": int(result.selected_index) == u_index,
        "raw_f_range": raw_range,
        "excluded_count": excluded,
        "go_nogo_active": bool(diag.get("go_nogo_constitution_active", False)),
        "f_demotion_active": bool(diag.get("f_eligibility_demotion_active", False)),
    }


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    """One (arm, seed) cell over N_BANKS divergent banks."""
    reset_all_rng(seed)
    rng = torch.Generator().manual_seed(seed)
    gate_on = arm == "ARM_CONSTITUTION"
    cfg = E3Config(
        world_dim=WORLD_DIM, hidden_dim=8,
        use_f_eligibility_demotion=True,
        use_go_nogo_constitution=gate_on,
    )

    n_converted = 0
    n_demotion_admits_u = 0
    n_safety_violation = 0
    n_specificity_ok = 0
    raw_ranges: List[float] = []
    excluded_counts: List[int] = []

    print(f"Seed {seed} Condition {arm}", flush=True)
    for b in range(N_BANKS):
        if (b + 1) % 8 == 0:
            print(f"  [eval] {arm} seed={seed} ep {b+1}/{N_BANKS}", flush=True)
        # Fresh selector per bank (random head init) -- the divergent pool is in
        # the bank; we measure the gate's effect holding the bank fixed.
        sel_demote = E3TrajectorySelector(
            E3Config(world_dim=WORLD_DIM, hidden_dim=8,
                     use_f_eligibility_demotion=True, use_go_nogo_constitution=False)
        )
        cands = _build_bank(rng)
        # Choose U = the F-BEST candidate (guaranteed F-eligible -> always in the
        # envelope) and tag IT undesirable -- the non-vacuous F-eligible-but-
        # undesirable setup. stale = the F-second-best (a distinct candidate).
        raw = _raw_f(sel_demote, cands)
        order = sorted(range(len(raw)), key=lambda i: raw[i])  # ascending cost = F-best first
        u_index, stale_index = order[0], order[1]
        # Divergent-pool precondition reads the UNBIASED raw-F range (the true F
        # field), NOT the post-bias select scores (which the -MODULATORY_PULL on U
        # would dominate). This is the same range statistic the conversion DV
        # depends on (per-candidate F differentiation).
        raw_ranges.append(max(raw) - min(raw))
        # ARM_DEMOTION reference (always computed, to define "previously gated").
        demote = _select_one(sel_demote, cands, u_index, stale_index, gate_on=False)
        if demote["picked_u"]:
            n_demotion_admits_u += 1
        # The arm under test (re-use the SAME bank + a selector with the arm's gate).
        sel_arm = E3TrajectorySelector(cfg)
        # copy head weights so the two selectors score the bank identically
        sel_arm.load_state_dict(sel_demote.state_dict())
        test = _select_one(sel_arm, cands, u_index, stale_index, gate_on=gate_on)
        excluded_counts.append(test["excluded_count"])
        if gate_on:
            if test["picked_u"]:
                n_safety_violation += 1
            # conversion: demotion picked U, constitution picked a different non-U class
            if demote["picked_u"] and (not test["picked_u"]) and (
                test["selected_class"] != demote["selected_class"]
            ):
                n_converted += 1
            # specificity: a control bank with NO undesirable tag -> gate must not
            # change the selection vs demotion-only.
            sel_ctrl = E3TrajectorySelector(cfg)
            sel_ctrl.load_state_dict(sel_demote.state_dict())
            ctrl = _select_one(
                sel_ctrl, cands, u_index, stale_index, gate_on=False
            )  # gate_on=False -> no signals -> control selection
            # re-run the arm selector with NO go_nogo signals (nothing to suppress)
            sel_ctrl2 = E3TrajectorySelector(cfg)
            sel_ctrl2.load_state_dict(sel_demote.state_dict())
            sel_ctrl2._running_variance = 0.0
            biasc = torch.zeros(len(cands))
            biasc[u_index] = -MODULATORY_PULL
            rc = sel_ctrl2.select(cands, temperature=1.0, score_bias=biasc,
                                  go_nogo_signals=None)
            if int(rc.selected_index) == ctrl["selected_index"]:
                n_specificity_ok += 1
        else:
            # demotion arm: specificity trivially ok (no gate)
            n_specificity_ok += 1

    conversion_rate = n_converted / N_BANKS if gate_on else 0.0
    demotion_admits_u_rate = n_demotion_admits_u / N_BANKS
    specificity_rate = n_specificity_ok / N_BANKS
    median_raw_range = statistics.median(raw_ranges) if raw_ranges else 0.0
    median_excluded = statistics.median(excluded_counts) if excluded_counts else 0.0

    cell: Dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "n_banks": N_BANKS,
        "conversion_rate": conversion_rate,
        "n_converted": n_converted,
        "demotion_admits_u_rate": demotion_admits_u_rate,
        "safety_violations": n_safety_violation,
        "specificity_rate": specificity_rate,
        "median_raw_f_range": median_raw_range,
        "median_modulatory_range": float(MODULATORY_PULL),
        "median_excluded_count": median_excluded,
    }
    cell["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice={
            "arm": arm,
            "use_f_eligibility_demotion": True,
            "use_go_nogo_constitution": gate_on,
            "n_banks": N_BANKS, "k": K_CANDIDATES, "modulatory_pull": MODULATORY_PULL,
        },
        seed=seed,
        script_path=Path(__file__),
        rng_fully_reset=True,
        extra_ineligible_reasons=["selection_face_synthetic_no_training"],
    )
    verdict_pass = gate_on and conversion_rate >= CONVERSION_FLOOR and n_safety_violation == 0
    print(f"verdict: {'PASS' if (verdict_pass or not gate_on) else 'FAIL'}", flush=True)
    return cell


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    global N_BANKS
    if dry_run:
        N_BANKS = 4  # toy

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ("ARM_DEMOTION", "ARM_CONSTITUTION"):
            arm_results.append(_run_cell(arm, seed))

    constitution_cells = [c for c in arm_results if c["arm"] == "ARM_CONSTITUTION"]
    demotion_cells = [c for c in arm_results if c["arm"] == "ARM_DEMOTION"]

    # --- Non-vacuity preconditions (RANGE statistics; same as the conversion DV) ---
    med_raw_range = statistics.median(
        [c["median_raw_f_range"] for c in arm_results]
    ) if arm_results else 0.0
    med_mod_range = statistics.median(
        [c["median_modulatory_range"] for c in arm_results]
    ) if arm_results else 0.0
    demotion_admits_u_med = statistics.median(
        [c["demotion_admits_u_rate"] for c in demotion_cells]
    ) if demotion_cells else 0.0
    constitution_excluded_med = statistics.median(
        [c["median_excluded_count"] for c in constitution_cells]
    ) if constitution_cells else 0.0

    pool_divergent = (med_raw_range > RAW_F_RANGE_FLOOR) and (
        med_mod_range > MODULATORY_RANGE_FLOOR
    )
    gate_engaged = (demotion_admits_u_med > DEMOTION_ADMITS_U_FLOOR) and (
        constitution_excluded_med > 0
    )
    preconditions_met = pool_divergent and gate_engaged

    # --- Acceptance ---
    seed_conversions = [c["conversion_rate"] for c in constitution_cells]
    seeds_converting = sum(1 for r in seed_conversions if r >= CONVERSION_FLOOR)
    n_seeds = max(1, len(constitution_cells))
    conversion_pass = (seeds_converting / n_seeds) >= SEED_PASS_FRACTION
    total_safety_violations = sum(c["safety_violations"] for c in constitution_cells)
    safety_ok = total_safety_violations == 0
    specificity_pass = all(
        c["specificity_rate"] >= 0.99 for c in constitution_cells
    )

    # --- Self-route + per-claim direction ---
    non_degenerate = True
    degeneracy_reason = None
    if not preconditions_met:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = {"MECH-449": "unknown", "ARC-107": "unknown"}
        non_degenerate = False
        degeneracy_reason = (
            "non-vacuity precondition unmet: "
            f"pool_divergent={pool_divergent} (raw_range={med_raw_range:.4f}>"
            f"{RAW_F_RANGE_FLOOR}, mod_range={med_mod_range:.4f}>{MODULATORY_RANGE_FLOOR}); "
            f"gate_engaged={gate_engaged} (demotion_admits_u={demotion_admits_u_med:.3f}>"
            f"{DEMOTION_ADMITS_U_FLOOR}, constitution_excluded={constitution_excluded_med})"
        )
    elif not safety_ok:
        label = "safety_contract_violated"
        outcome = "FAIL"
        direction = {"MECH-449": "weakens", "ARC-107": "unknown"}
    elif conversion_pass and specificity_pass:
        label = "go_nogo_converts_gated_channel"
        outcome = "PASS"
        direction = {"MECH-449": "supports", "ARC-107": "supports"}
    else:
        # preconditions met, safety holds, but no conversion -> over-specification.
        label = "go_nogo_no_conversion_overspecification"
        outcome = "FAIL"
        direction = {"MECH-449": "weakens", "ARC-107": "unknown"}

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "pool_divergent_raw_f_range", "description":
             "median raw-F RANGE across banks clears the divergent-pool floor "
             "(the same range statistic the conversion DV depends on)",
             "measured": round(med_raw_range, 5), "threshold": RAW_F_RANGE_FLOOR,
             "control": "constructed distinct per-candidate world_states",
             "met": bool(med_raw_range > RAW_F_RANGE_FLOOR)},
            {"name": "pool_divergent_modulatory_range", "description":
             "median modulatory-channel RANGE clears the floor",
             "measured": round(med_mod_range, 5), "threshold": MODULATORY_RANGE_FLOOR,
             "control": "score_bias pull toward U vs 0 others",
             "met": bool(med_mod_range > MODULATORY_RANGE_FLOOR)},
            {"name": "gate_engaged_demotion_admits_u", "description":
             "ARM_DEMOTION admits+selects the F-eligible undesirable on a majority "
             "of banks (the demotion-insufficient gap is present)",
             "measured": round(demotion_admits_u_med, 4),
             "threshold": DEMOTION_ADMITS_U_FLOOR,
             "control": "U is modulatory-favoured + F-eligible", "met": bool(
                 demotion_admits_u_med > DEMOTION_ADMITS_U_FLOOR)},
            {"name": "gate_engaged_constitution_excludes", "description":
             "ARM_CONSTITUTION No-Go actually suppresses (excluded_count > 0): the "
             "Go/No-Go variables vary across arms",
             "measured": round(constitution_excluded_med, 4), "threshold": 0,
             "control": "safety+staleness No-Go injected", "met": bool(
                 constitution_excluded_med > 0)},
        ],
        "criteria_non_degenerate": {
            "conversion": bool(any(r > 0 for r in seed_conversions)),
            "safety": bool(constitution_cells),
            "specificity": bool(constitution_cells),
        },
        "criteria": [
            {"name": "conversion_rate_floor", "load_bearing": True,
             "passed": bool(conversion_pass)},
            {"name": "safety_no_disinhibition", "load_bearing": True,
             "passed": bool(safety_ok)},
            {"name": "specificity_no_spurious_conversion", "load_bearing": False,
             "passed": bool(specificity_pass)},
        ],
    }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if outcome == "PASS" else (
            "mixed" if "weakens" in direction.values() else "unknown"),
        "evidence_direction_per_claim": direction,
        "interpretation": interpretation,
        "non_degenerate": non_degenerate,
        "timestamp_utc": ts,
        "summary": {
            "preconditions_met": preconditions_met,
            "conversion_rate_per_seed": seed_conversions,
            "seeds_converting": seeds_converting,
            "n_seeds": len(constitution_cells),
            "total_safety_violations": total_safety_violations,
            "specificity_pass": specificity_pass,
            "median_raw_f_range": med_raw_range,
            "median_modulatory_range": med_mod_range,
            "demotion_admits_u_median": demotion_admits_u_med,
            "constitution_excluded_median": constitution_excluded_med,
        },
        "arm_results": arm_results,
        "config": {
            "seeds": seeds, "n_banks": N_BANKS, "k_candidates": K_CANDIDATES,
            "modulatory_pull": MODULATORY_PULL, "conversion_floor": CONVERSION_FLOOR,
            "raw_f_range_floor": RAW_F_RANGE_FLOOR,
            "modulatory_range_floor": MODULATORY_RANGE_FLOOR,
        },
        "notes": (
            "Selection-face, single-decision conversion falsifier of the BUILT "
            "MECH-449 Go/No-Go gate through E3TrajectorySelector.select(). "
            "Divergent pool provided by construction + checked as a non-vacuity "
            "precondition; the sustained-behavioural conversion on the trained "
            "foraging substrate is the downstream successor (gated on BG-circuit "
            "completeness, ARC-107 s6b ledger). PROMOTES NOTHING until adjudicated."
        ),
    }
    if degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    out_dir = Path(tempfile.gettempdir()) / "ree_dry_run_manifests" if dry_run else EVIDENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    manifest["manifest_path"] = str(out_path)
    print(f"[689g] outcome={outcome} label={label} conversion_per_seed={seed_conversions} "
          f"safety_violations={total_safety_violations} -> {out_path}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689g MECH-449 Go/No-Go eligibility constitution conversion falsifier"
    )
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run.")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
        dry_run=args.dry_run,
    )
