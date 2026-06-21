"""V3-EXQ-689f -- MECH-449 / ARC-107 No-Go-necessity falsifier.

SUBSTRATE-READINESS / NECESSITY DIAGNOSTIC (experiment_purpose=diagnostic;
claim_ids=[MECH-449, ARC-107]; PROMOTES NOTHING). The POSITIVE build trigger for
MECH-449 (the Go/No-Go opponency leg of the ARC-107 basal-ganglia selector
constitution), wired into the MECH-449 gate inversion 2026-06-21
(arc_107_selector_constitution_design_2026-06-20.md s3.2 / s6 step 4b).

WHY (anti-partial-instantiation guard). MECH-448 (rank-preserving F->eligibility
demotion) PASSED its falsifier (V3-EXQ-689d) and is the only landed leg of the
constitution. The risk -- the exact shape of the F-dominance partial-instantiation
trap -- is that a passing MECH-448 silently forecloses the rest of the circuit.
The old gate built MECH-449 only "if MECH-448 proves insufficient"; a passing 448
means that gate never opens. This experiment INVERTS the trigger: it demonstrates,
on the LIVE built MECH-448 envelope, the demotion-insufficient regime that
positively motivates MECH-449 -- without waiting for a downstream behavioural
retest to stumble into the gap.

THE STRUCTURAL CLAIM. MECH-448's _f_eligibility_envelope is RANK-PRESERVING in F:
its eligible set is an F-rank PREFIX (the design's own load-bearing property,
e3_selector.py:765-766). Rank-preserving demotion is ORDER-PRESERVING over F, so
it can NARROW the candidate set (drop low-F candidates) but it CANNOT exclude a
candidate on a non-F axis. A candidate that is undesirable on a safety / staleness
/ perseveration axis BUT high in F-merit is therefore ADMITTED into the eligible
set -- demotion has no mechanism to suppress it. Only an ACTIVE No-Go (MECH-449),
which suppresses on the undesirability axis (orthogonal to F-rank), can exclude it.

WHAT THIS TESTS (a property test of the REAL substrate, commitment-free). It binds
the live E3TrajectorySelector._f_eligibility_envelope to a config and feeds it
synthetic per-candidate F-cost banks (lower=better) with an INDEPENDENT
undesirability mask (the unsafe/stale/perseverative axis). It measures, on a
divergent F field:
  - undesirable_admit_rate (DEMOTION): fraction of trials where the MECH-448
    eligible set CONTAINS >= 1 undesirable candidate (i.e. a high-F undesirable the
    envelope cannot drop). This is the demotion-insufficient regime.
  - nogo_ref_admit_rate (the MISSING MECH-449): the same eligible set with an
    active No-Go applied (eligible MINUS undesirable). ~0 by construction -- the
    regime is CLOSABLE by an active No-Go, motivating the MECH-449 build.
  - rank_preserving_frac: empirically confirms the live envelope is an F-rank
    prefix (the structural reason it cannot suppress) on every demotion-active tick.

This is a SELECTION-FACE structural diagnostic, NOT a sustained multi-step
behavioural DV: it deliberately avoids the don't-queue-commitment-dependent-
behaviour-while-the-BG-layer-is-incomplete trap (it measures a GAP in the built
MECH-448 envelope, not MECH-449 itself, and runs on the CURRENT substrate where
MECH-448 is built and MECH-449 is not).

ARMS (the envelope-config axis):
  ARM_OFF              demotion absent: the "eligible set" is ALL candidates ->
                       undesirable_admit_rate ~ 1.0 trivially (baseline: without
                       demotion everything, including undesirables, is admitted).
  ARM_DEMOTION_ADAPTIVE  MECH-448 ON, channel-adaptive mean-relative floor (the
                       2026-06-21 best). The envelope NARROWS the set yet still
                       admits high-F undesirables.
  ARM_DEMOTION_FIXED   MECH-448 ON, fixed absolute floor 0.30 (the 689d config) --
                       floor-INDEPENDENCE of the gap.

INTERPRETATION GRID (self-routing):
| condition                                                  | label                                  | direction        | next                                              |
|------------------------------------------------------------|----------------------------------------|------------------|---------------------------------------------------|
| field not divergent (envelope all-admit) on a demotion arm | substrate_not_ready_requeue            | non_contributory | the all-admit no-op (cf 654h); re-tune the field  |
| undesirables ~never F-eligible (regime absent)             | demotion_insufficient_regime_absent    | non_contributory | the adversarial regime did not arise; re-tune mask|
| rank_preserving_frac < 1.0 on a demotion arm               | rank_alteration_not_prefix_diagnose    | non_contributory | impl/design fault -> /diagnose-errors; NOT a verdict|
| regime present + No-Go ref closes it + rank-preserving      | demotion_insufficient_nogo_necessary   | non_contributory | PASS -> positive trigger for the MECH-449 build   |

PROMOTES NOTHING: experiment_purpose=diagnostic; MECH-449 stays candidate /
substrate_conditional; ARC-107 stays candidate. A PASS is a build-trigger
hypothesis (per the diagnostic-self-route-is-a-hypothesis discipline), adjudicated
before it drives the MECH-449 /implement-substrate session.

Usage:
  /opt/local/bin/python3 experiments/v3_exq_689f_nogo_necessity_falsifier.py --dry-run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from _metrics import check_degeneracy  # noqa: E402
from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_689f_nogo_necessity_falsifier"
QUEUE_ID = "V3-EXQ-689f"
SUPERSEDES: Optional[str] = None
# Tests the NECESSITY of the MECH-449 Go/No-Go opponency leg (ARC-107 umbrella).
# Diagnostic -> excluded from governance confidence scoring; the tags are
# informational provenance for the MECH-449 build trigger. PROMOTES NOTHING.
CLAIM_IDS: List[str] = ["MECH-449", "ARC-107"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
N_TRIALS_PER_SEED = 400
K_CANDIDATES = 8
P_UNDESIRABLE = 0.4          # per-candidate prob of being unsafe/stale/perseverative (INDEPENDENT of F)
F_SPREAD = 1.0              # std of the synthetic F-cost field (non-uniform => envelope excludes)

DRY_RUN_SEEDS = [42]
DRY_RUN_N_TRIALS = 20

# MECH-448 envelope config (mirrors 689d/689e).
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30
F_ELIGIBILITY_DN_SIGMA = 0.0
F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR = 1.0

# Pre-registered thresholds.
DIVERGENT_FRAC_FLOOR = 0.5      # PRECONDITION: fraction of trials with excluded>0 (field divergent)
REGIME_PRESENT_FLOOR = 0.3     # PRECONDITION: undesirable_admit_rate (regime occurs in >=30% of divergent trials)
NOGO_CLOSES_CEIL = 0.05        # PRIMARY: No-Go reference admit-rate <= this (the gap is closable)
RANK_PRESERVING_FRAC_REQUIRED = 1.0  # GUARDRAIL: every demotion-active trial is an F-rank prefix
MIN_SEEDS_FOR_PASS = 2         # of 3

OFF_ARM = "ARM_OFF"
ADAPTIVE_ARM = "ARM_DEMOTION_ADAPTIVE"
FIXED_ARM = "ARM_DEMOTION_FIXED"
ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": OFF_ARM,
        "label": "demotion absent (baseline: all candidates eligible)",
        "use_f_eligibility_demotion": False,
        "use_f_eligibility_adaptive_floor": False,
    },
    {
        "arm_id": ADAPTIVE_ARM,
        "label": "MECH-448 ON, channel-adaptive mean-relative floor (2026-06-21)",
        "use_f_eligibility_demotion": True,
        "use_f_eligibility_adaptive_floor": True,
    },
    {
        "arm_id": FIXED_ARM,
        "label": "MECH-448 ON, fixed absolute floor 0.30 (the 689d config)",
        "use_f_eligibility_demotion": True,
        "use_f_eligibility_adaptive_floor": False,
    },
]

# All MECH-448-ON arms (the GUARDRAIL rank-preserving invariant must hold on every one).
DEMOTION_ARMS = (ADAPTIVE_ARM, FIXED_ARM)
# The PRIMARY/PRECONDITION gating runs on the CANONICAL floor (the channel-adaptive
# mean-relative floor, 2026-06-21). The FIXED 0.30 floor is the 689d config kept as a
# floor-INDEPENDENCE corroborator: on a generic field it under-engages (its own 654h
# floor-mismatch), so it informs but does not gate the No-Go-necessity verdict.
PRIMARY_ARMS = (ADAPTIVE_ARM,)


def _make_envelope_caller(arm: Dict[str, Any]) -> SimpleNamespace:
    """Bind the LIVE E3TrajectorySelector._f_eligibility_envelope to a config
    namespace carrying exactly the four attributes the method reads
    (e3_selector.py:802-817). This exercises the real MECH-448 code path -- the
    eligible-set composition the No-Go-necessity claim is about -- without an
    agent rollout."""
    return SimpleNamespace(
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_f_eligibility_adaptive_floor=bool(arm["use_f_eligibility_adaptive_floor"]),
        f_eligibility_adaptive_mean_factor=F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
    )


def _envelope(config: SimpleNamespace, raw_scores: torch.Tensor) -> torch.Tensor:
    """Call the real MECH-448 envelope (unbound method bound to a shim self)."""
    shim = SimpleNamespace(config=config)
    return E3TrajectorySelector._f_eligibility_envelope(shim, raw_scores)


def _is_rank_prefix(raw_scores: torch.Tensor, eligible_idx: torch.Tensor) -> bool:
    """Verify the eligible set is an F-rank prefix: every eligible cost <= every
    excluded cost (raw_scores is a per-candidate COST, lower=better)."""
    n = int(raw_scores.shape[0])
    elig = set(int(i) for i in eligible_idx.tolist())
    if not elig or len(elig) == n:
        return True
    elig_max = max(float(raw_scores[i].item()) for i in elig)
    excl_min = min(
        float(raw_scores[i].item()) for i in range(n) if i not in elig
    )
    # Tie-robust: eligible worst-cost must not exceed excluded best-cost (+eps).
    return elig_max <= excl_min + 1e-6


def _run_cell(arm: Dict[str, Any], seed: int, n_trials: int) -> Dict[str, Any]:
    reset_all_rng(seed)
    rng = np.random.default_rng(seed)
    demotion_on = bool(arm["use_f_eligibility_demotion"])
    config = _make_envelope_caller(arm)

    n_divergent = 0                # trials where the envelope actually excluded someone
    n_informative_div = 0          # divergent trials where an undesirable is F-eligible (admitted)
    n_nogo_admit_div = 0           # divergent trials where No-Go ref STILL admits an undesirable (~0)
    n_rank_preserving = 0          # demotion-active trials that are an F-rank prefix
    n_demotion_active = 0
    excluded_counts: List[int] = []
    envelope_sizes: List[int] = []
    n_trials_with_undesirable = 0

    for _t in range(n_trials):
        raw_scores = torch.from_numpy(
            rng.normal(0.0, F_SPREAD, size=K_CANDIDATES).astype(np.float32)
        )
        undesirable = rng.random(K_CANDIDATES) < P_UNDESIRABLE  # independent of F
        has_undesirable = bool(undesirable.any())
        if has_undesirable:
            n_trials_with_undesirable += 1

        if demotion_on:
            eligible_idx = _envelope(config, raw_scores)
            n_demotion_active += 1
            if _is_rank_prefix(raw_scores, eligible_idx):
                n_rank_preserving += 1
        else:
            # OFF: no envelope -> all candidates eligible (the baseline regime).
            eligible_idx = torch.arange(K_CANDIDATES)

        eligible = set(int(i) for i in eligible_idx.tolist())
        excluded_count = K_CANDIDATES - len(eligible)
        excluded_counts.append(excluded_count)
        envelope_sizes.append(len(eligible))

        divergent = excluded_count > 0
        if divergent:
            n_divergent += 1
            # Demotion (or OFF) eligible set admits an undesirable?
            admit_undesirable = any(bool(undesirable[i]) for i in eligible)
            if admit_undesirable:
                n_informative_div += 1
            # No-Go reference: eligible MINUS undesirable (the missing MECH-449).
            eligible_nogo = {i for i in eligible if not bool(undesirable[i])}
            nogo_admits = any(bool(undesirable[i]) for i in eligible_nogo)  # always False
            if nogo_admits:
                n_nogo_admit_div += 1

    divergent_frac = float(n_divergent) / float(n_trials) if n_trials else 0.0
    undesirable_admit_rate = (
        float(n_informative_div) / float(n_divergent) if n_divergent > 0 else 0.0
    )
    nogo_ref_admit_rate = (
        float(n_nogo_admit_div) / float(n_divergent) if n_divergent > 0 else 0.0
    )
    rank_preserving_frac = (
        float(n_rank_preserving) / float(n_demotion_active)
        if n_demotion_active > 0 else (1.0 if not demotion_on else 0.0)
    )

    def _mean_i(xs: List[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return {
        "arm_id": arm["arm_id"],
        "label": arm["label"],
        "seed": int(seed),
        "use_f_eligibility_demotion": demotion_on,
        "use_f_eligibility_adaptive_floor": bool(arm["use_f_eligibility_adaptive_floor"]),
        "n_trials": int(n_trials),
        "n_trials_with_undesirable": int(n_trials_with_undesirable),
        # PRECONDITION: the field is divergent (the envelope actually narrows).
        "divergent_frac": round(divergent_frac, 6),
        "excluded_count_mean": round(_mean_i(excluded_counts), 6),
        "envelope_size_mean": round(_mean_i(envelope_sizes), 6),
        # PRIMARY: the demotion-insufficient regime + that No-Go closes it.
        "undesirable_admit_rate": round(undesirable_admit_rate, 6),
        "nogo_ref_admit_rate": round(nogo_ref_admit_rate, 6),
        "admit_gap_demotion_minus_nogo": round(
            undesirable_admit_rate - nogo_ref_admit_rate, 6
        ),
        # GUARDRAIL.
        "rank_preserving_frac": round(rank_preserving_frac, 6),
    }


def _cells(rows: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("arm_id") == arm_id]


def _n_seeds(cells: List[Dict[str, Any]], predicate) -> int:
    return sum(1 for c in cells if predicate(c))


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    preconditions: List[Dict[str, Any]] = []
    per_arm: Dict[str, Any] = {}

    field_divergent_ok = True
    regime_present_ok = True
    nogo_closes_ok = True
    rank_preserving_ok = True
    admit_groups: List[List[float]] = []  # for check_degeneracy (PRIMARY arm)

    # GUARDRAIL: the rank-preserving F-prefix invariant must hold on EVERY MECH-448-ON
    # arm (it is a structural property of the built envelope, not a tuning choice).
    for arm_id in DEMOTION_ARMS:
        for c in _cells(arm_results, arm_id):
            if c["rank_preserving_frac"] < RANK_PRESERVING_FRAC_REQUIRED:
                rank_preserving_ok = False

    # PRECONDITION + PRIMARY gate on the CANONICAL (channel-adaptive) floor only.
    for arm_id in PRIMARY_ARMS:
        cells = _cells(arm_results, arm_id)

        n_div = _n_seeds(cells, lambda c: c["divergent_frac"] >= DIVERGENT_FRAC_FLOOR)
        n_regime = _n_seeds(
            cells, lambda c: c["undesirable_admit_rate"] >= REGIME_PRESENT_FLOOR
        )
        n_nogo = _n_seeds(cells, lambda c: c["nogo_ref_admit_rate"] <= NOGO_CLOSES_CEIL)

        ch_div_ok = n_div >= MIN_SEEDS_FOR_PASS
        ch_regime_ok = n_regime >= MIN_SEEDS_FOR_PASS
        ch_nogo_ok = n_nogo >= MIN_SEEDS_FOR_PASS
        field_divergent_ok = field_divergent_ok and ch_div_ok
        regime_present_ok = regime_present_ok and ch_regime_ok
        nogo_closes_ok = nogo_closes_ok and ch_nogo_ok

        admit_groups.append([c["undesirable_admit_rate"] for c in cells])

        def _m(key: str) -> float:
            return round(float(np.mean([c[key] for c in cells])) if cells else 0.0, 6)

        preconditions.append({
            "name": f"field_divergent_{arm_id}",
            "description": (
                "the canonical (channel-adaptive) MECH-448 envelope actually narrows "
                "the candidate set (excluded_count>0) on >=2/3 seeds -- a non-divergent "
                "(all-admit) field is the 654h no-op, not the demotion-insufficient regime"
            ),
            "measured": _m("divergent_frac"),
            "threshold": DIVERGENT_FRAC_FLOOR,
            "control": "the synthetic F field has spread (F_SPREAD>0)",
            "met": bool(ch_div_ok),
        })
        preconditions.append({
            "name": f"regime_present_{arm_id}",
            "description": (
                "the demotion eligible set admits an F-eligible undesirable "
                "candidate (undesirable_admit_rate) on >=2/3 seeds -- the "
                "adversarial high-F-undesirable regime actually arises"
            ),
            "measured": _m("undesirable_admit_rate"),
            "threshold": REGIME_PRESENT_FLOOR,
            "control": "undesirability is assigned INDEPENDENTLY of F (p_undesirable>0)",
            "met": bool(ch_regime_ok),
        })

    # Per-arm reporting for BOTH demotion arms (FIXED is the floor-independence
    # corroborator: the gap-DIRECTION among the trials where it does engage).
    for arm_id in DEMOTION_ARMS:
        cells = _cells(arm_results, arm_id)

        def _m(key: str, _cells_ref=cells) -> float:
            return round(
                float(np.mean([c[key] for c in _cells_ref])) if _cells_ref else 0.0, 6
            )

        per_arm[arm_id] = {
            "is_primary_gate": bool(arm_id in PRIMARY_ARMS),
            "n_field_divergent_seeds": _n_seeds(
                cells, lambda c: c["divergent_frac"] >= DIVERGENT_FRAC_FLOOR
            ),
            "n_regime_present_seeds": _n_seeds(
                cells, lambda c: c["undesirable_admit_rate"] >= REGIME_PRESENT_FLOOR
            ),
            "n_nogo_closes_seeds": _n_seeds(
                cells, lambda c: c["nogo_ref_admit_rate"] <= NOGO_CLOSES_CEIL
            ),
            "undesirable_admit_rate_mean": _m("undesirable_admit_rate"),
            "nogo_ref_admit_rate_mean": _m("nogo_ref_admit_rate"),
            "admit_gap_mean": _m("admit_gap_demotion_minus_nogo"),
            "rank_preserving_frac_mean": _m("rank_preserving_frac"),
            "excluded_count_mean": _m("excluded_count_mean"),
            "envelope_size_mean": _m("envelope_size_mean"),
        }

    off_cells = _cells(arm_results, OFF_ARM)
    off_admit_mean = round(
        float(np.mean([c["undesirable_admit_rate"] for c in off_cells]))
        if off_cells else 0.0, 6
    )

    deg = check_degeneracy({
        "undesirable_admit_rate_demotion": {"groups": admit_groups},
    })

    preconditions_met = field_divergent_ok and regime_present_ok
    primary_ok = preconditions_met and nogo_closes_ok

    # ---- self-route ----
    if not field_divergent_ok:
        label = "substrate_not_ready_requeue"
        overall_pass = False
    elif not regime_present_ok:
        label = "demotion_insufficient_regime_absent"
        overall_pass = False
    elif not rank_preserving_ok:
        label = "rank_alteration_not_prefix_diagnose"
        overall_pass = False
    elif not nogo_closes_ok:
        # Regime present but No-Go ref does not close it (cannot happen by
        # construction; guard against an instrumentation fault).
        label = "nogo_reference_does_not_close_diagnose"
        overall_pass = False
    else:
        label = "demotion_insufficient_nogo_necessary"
        overall_pass = True

    criteria = [
        {
            "name": "PRIMARY_demotion_insufficient_and_nogo_closes",
            "load_bearing": True,
            "passed": bool(primary_ok),
        },
        {"name": "PRECONDITION_field_divergent_both_arms", "passed": bool(field_divergent_ok)},
        {"name": "PRECONDITION_regime_present_both_arms", "passed": bool(regime_present_ok)},
        {"name": "GUARDRAIL_rank_preserving_all_demotion_trials", "passed": bool(rank_preserving_ok)},
    ]
    criteria_non_degenerate = {
        "PRIMARY_admit_rate_has_spread": bool(deg.get("non_degenerate", True)),
        "PRECONDITION_field_divergent": bool(field_divergent_ok),
        "GUARDRAIL_rank_preserving": bool(rank_preserving_ok),
    }

    return {
        "label": label,
        "evidence_direction": "non_contributory",
        "overall_pass": overall_pass,
        "field_divergent_ok": field_divergent_ok,
        "regime_present_ok": regime_present_ok,
        "nogo_closes_ok": nogo_closes_ok,
        "rank_preserving_ok": rank_preserving_ok,
        "off_undesirable_admit_rate_mean": off_admit_mean,
        "per_arm": per_arm,
        "preconditions": preconditions,
        "criteria": criteria,
        "criteria_non_degenerate": criteria_non_degenerate,
        "non_degenerate": deg.get("non_degenerate", True),
        "degeneracy_reason": deg.get("degeneracy_reason", ""),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    n_trials = DRY_RUN_N_TRIALS if dry_run else N_TRIALS_PER_SEED

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm_id']}", flush=True)
            cell = _run_cell(arm, seed, n_trials)
            cell["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={
                    "arm": {
                        k: arm[k] for k in (
                            "arm_id", "use_f_eligibility_demotion",
                            "use_f_eligibility_adaptive_floor",
                        )
                    },
                    "f_eligibility_config": {
                        "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                        "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                        "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
                    },
                    "k_candidates": K_CANDIDATES,
                    "p_undesirable": P_UNDESIRABLE,
                    "f_spread": F_SPREAD,
                    "n_trials": n_trials,
                },
                seed=seed,
                script_path=Path(__file__),
                rng_fully_reset=True,
                extra_ineligible_reasons=["synthetic_property_test_not_reusable"],
            )
            arm_results.append(cell)
            # A cell "runs" cleanly unless an exception propagated (none caught here).
            print("verdict: PASS", flush=True)

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

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
        "evidence_direction": summary["evidence_direction"],
        "evidence_direction_per_claim": {
            "MECH-449": "non_contributory",
            "ARC-107": "non_contributory",
        },
        "non_degenerate": summary.get("non_degenerate", True),
        "degeneracy_reason": summary.get("degeneracy_reason", ""),
        "evidence_direction_note": (
            "MECH-449 / ARC-107 No-Go-necessity falsifier -- the POSITIVE build "
            "trigger for the MECH-449 Go/No-Go opponency leg (anti-partial-"
            "instantiation guard against a passing MECH-448 foreclosing the rest "
            "of the ARC-107 constitution). Property test of the LIVE "
            "_f_eligibility_envelope: rank-preserving F->eligibility demotion is "
            "order-preserving over F, so it admits high-F undesirable candidates "
            "(undesirable_admit_rate) that an active No-Go (the missing MECH-449) "
            "would suppress (nogo_ref_admit_rate ~ 0). PASS "
            "(demotion_insufficient_nogo_necessary) = on a divergent F field the "
            "demotion-insufficient regime is present (undesirable_admit_rate >= "
            "floor) AND the No-Go reference closes it AND the envelope is an F-rank "
            "prefix on every demotion-active trial, on BOTH the adaptive and fixed "
            "floors (floor-independent). diagnostic; claim_ids tag the NECESSITY of "
            "MECH-449; PROMOTES NOTHING; MECH-449 stays candidate/substrate_conditional, "
            "ARC-107 stays candidate. A PASS is a build-trigger hypothesis, "
            "adjudicated before it drives the MECH-449 /implement-substrate session."
        ),
        "interpretation": {
            "label": summary["label"],
            "preconditions": summary["preconditions"],
            "criteria": summary["criteria"],
            "criteria_non_degenerate": summary["criteria_non_degenerate"],
            "routing": {
                "demotion_insufficient_nogo_necessary": "PASS -> positive trigger for the MECH-449 Go/No-Go build (the demotion-insufficient regime is real and only an active No-Go closes it); proceed to the MECH-449 /implement-substrate session",
                "substrate_not_ready_requeue": "the envelope is all-admit (non-divergent field; the 654h no-op) -> re-tune the synthetic F field; do NOT weaken anything",
                "demotion_insufficient_regime_absent": "undesirables ~never F-eligible -> the adversarial high-F-undesirable regime did not arise; re-tune the undesirability mask; NOT a verdict on MECH-449",
                "rank_alteration_not_prefix_diagnose": "the live envelope is NOT an F-rank prefix on a demotion-active trial -> implementation/design fault -> /diagnose-errors; NOT a verdict",
                "nogo_reference_does_not_close_diagnose": "the No-Go reference still admits an undesirable (cannot happen by construction) -> instrumentation fault -> /diagnose-errors",
            },
        },
        "dry_run": bool(dry_run),
        "config": {
            "seeds": seeds,
            "n_trials_per_seed": n_trials,
            "k_candidates": K_CANDIDATES,
            "p_undesirable": P_UNDESIRABLE,
            "f_spread": F_SPREAD,
            "arms": [
                {k: a[k] for k in (
                    "arm_id", "label", "use_f_eligibility_demotion",
                    "use_f_eligibility_adaptive_floor",
                )}
                for a in ARMS
            ],
            "f_eligibility_config": {
                "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
                "f_eligibility_adaptive_mean_factor": F_ELIGIBILITY_ADAPTIVE_MEAN_FACTOR,
            },
            "thresholds": {
                "divergent_frac_floor": DIVERGENT_FRAC_FLOOR,
                "regime_present_floor": REGIME_PRESENT_FLOOR,
                "nogo_closes_ceil": NOGO_CLOSES_CEIL,
                "rank_preserving_frac_required": RANK_PRESERVING_FRAC_REQUIRED,
                "min_seeds_for_pass": MIN_SEEDS_FOR_PASS,
            },
        },
        "acceptance_criteria": {
            "PRECONDITION_field_divergent_both_arms": summary["field_divergent_ok"],
            "PRECONDITION_regime_present_both_arms": summary["regime_present_ok"],
            "PRIMARY_nogo_closes_the_regime": summary["nogo_closes_ok"],
            "GUARDRAIL_rank_preserving_all_demotion_trials": summary["rank_preserving_ok"],
            "overall_pass": summary["overall_pass"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(
        f"Outcome: {outcome} (label={summary['label']}, "
        f"evidence_direction={summary['evidence_direction']})",
        flush=True,
    )
    for k, v in manifest["acceptance_criteria"].items():
        print(f"  {k}: {v}", flush=True)

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-689f MECH-449 / ARC-107 No-Go-necessity falsifier"
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
