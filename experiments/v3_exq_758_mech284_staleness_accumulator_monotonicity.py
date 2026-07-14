#!/opt/local/bin/python3
"""
V3-EXQ-758 -- MECH-284 V_s residual schema-staleness accumulator:
WALL-INDEPENDENT, action-free confirming test of region-indexed staleness tracking.

WHAT THIS IS. A read-only functional-signature test of the MECH-284 substrate
(ree_core/hippocampal/staleness_accumulator.py -> StalenessAccumulator). It drives
the REAL .integrate() / .tick_leak() code path with REAL BroadcastEvent
(ree_core/regulators/invalidation_trigger.py) and REAL Anchor
(ree_core/hippocampal/anchor_set.py) objects, then reads the region-indexed scalar
map via .snapshot(). No agent, no environment, no committed behaviour -- the DV is
the accumulator's own scalar state.

WHY WALL-INDEPENDENT (the point of this design). The V3 program is bottlenecked on
the competence wall (behavioral_diversity_isolation:GAP-I; in-flight V3-EXQ-752..756).
Any committed-behaviour DV is wall-bound. MECH-284's operational definition is a
region-indexed scalar accumulator whose state is directly readable, so its
staleness-tracking monotonicity / region-locality / leak are measurable WITHOUT any
action selection -- the reading passes or fails independent of the competence wall
(precedent: functional-signature DVs passed in V3-EXQ-455/447/448 while the
behavioural baseline was monostrategy-locked; failure_autopsy_455a). MECH-284 is
referenced by V3-EXQ-455a / V3-EXQ-592f but neither tested the accumulator directly;
exp_conf is 0.0. This run supplies the first direct V3 evidence.

MECH-284 OPERATIONAL DEFINITION (claims.yaml, refined 2026-04-22):
    For each schema region r in active_anchor_set(t):
        if MECH-287 trigger(t):
            staleness[r] += attribution_weight(r, source_streams) * magnitude
        staleness[r] *= leak_factor
  Region key = (scale, segment_id). Attribution modes: "equal" (uniform 1/N over
  active anchors) and "stream_overlap" (|source_sources & stream_mixture| /
  max(|source_sources|,1) per anchor). Clip at staleness_clip; drop below
  drop_epsilon.

DESIGN (three phases, per seed; all read-only arithmetic on the real module):
  Phase A -- GRADED ACCUMULATION + REGION-LOCALITY (attribution_mode=stream_overlap).
    Build N_REGIONS real Anchors with DISJOINT stream mixtures. ALL anchors are in
    the active set every step (the stringent locality test: fresh anchors are
    ACTIVE, not merely absent). Integrate L broadcasts whose source_sources overlap
    ONLY the perturbed region p (full overlap -> credit 1.0), each of strength DOSE,
    WITHOUT reset -> cumulative. Read snapshot() after each dose.
      DV(monotone rise): V_s[p] over the L doses is non-decreasing and rises by
        >= RISE_MARGIN (min(DOSE*k, clip)).
      DV(region-locality): fresh regions (disjoint streams) receive ZERO credit and
        stay <= LOCALITY_FLOOR while V_s[p] climbs; peak cross-region spread
        (V_s[p] - max_fresh) >= SPREAD_MARGIN. This is the LOAD-BEARING criterion
        (routes on cross-region RANGE/spread).
  Phase B -- LEAK DECAY. From the Phase-A peak, call tick_leak() LEAK_TICKS times.
      DV(leak): V_s[p] is monotone non-increasing, decays to <= LEAK_DECAY_FRAC of
        the peak, and the region drops out of snapshot() (below drop_epsilon).
  Phase C -- EQUAL-MODE MONOTONICITY CONTROL (attribution_mode=equal, active set =
    {perturbed anchor} only). Corroborates that the 'equal' credit branch also
    accumulates monotonically and that active-set control localises. REPORTED, not
    gating (keeps the verdict on the three headline DVs).

PASS  = C1 (monotone rise) AND C2 (region-locality spread) AND C3 (leak decay), on
        every seed, with the readiness gate met. -> supports MECH-284.
WEAKENS (readiness met, a headline criterion fails on any seed): the accumulator
        does not track staleness monotonically / localise / leak as specified.
        evidence_direction = weakens (routes /failure-autopsy; does not auto-demote).

NON-VACUITY / READINESS GATE (substrate_not_ready_requeue -- never a false weakens).
  ARMING CAVEAT (lesson from V3-EXQ-688's vacuous null): the accumulator is populated
  ONLY via the real .integrate() path with positive strength + genuine stream
  overlap -- NOT any default-1.0 per_stream_vs fallback. Before scoring, assert on
  the positive control (perturbed region after graded integrate() vs the fresh
  non-overlapping regions) that peak cross-region RANGE > RANGE_FLOOR -- the SAME
  statistic the load-bearing locality criterion C2 routes on (per the V3-EXQ-643
  same-statistic rule: a range-gated criterion needs a range readiness check, not a
  magnitude proxy). If the map is empty or flat across regions (range <= floor) the
  substrate did not populate -> outcome FAIL, evidence_direction non_contributory,
  non_degenerate false, interpretation.label substrate_not_ready_requeue. A genuine
  substrate regression thus routes non_contributory, never a false weakens.

Substrate is BUILT (StalenessAccumulator, HippocampalConfig.use_staleness_accumulator).
No substrate build owed; re-derive brake does not fire (0 substrate_ceiling /
non_contributory autopsies tag MECH-284). Ethics preflight: all-false / allow (V3
pre-ethical instrumentation; pure float arithmetic, no valence / self-model / harm).
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.hippocampal.anchor_set import Anchor  # noqa: E402
from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator  # noqa: E402
from ree_core.regulators.invalidation_trigger import BroadcastEvent  # noqa: E402
from ree_core.utils.config import StalenessAccumulatorConfig  # noqa: E402

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_758_mech284_staleness_accumulator_monotonicity"
CLAIM_IDS = ["MECH-284"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# ------------------------------------------------------------------ #
# Protocol constants (pre-registered)                                #
# ------------------------------------------------------------------ #
SCALE = "slow"
N_REGIONS = 4
L_LEVELS = 8            # graded accumulation doses (Phase A + Phase C)
DOSE = 0.15            # per-broadcast strength
LEAK_TICKS = 30        # tick_leak() calls in Phase B
LEAK_FACTOR = 0.5      # crisp geometric decay for a short probe
STALENESS_CLIP = 1.0
DROP_EPSILON = 1e-6

# episodes_per_run denominator = graded doses + leak ticks (per seed).
STEPS_PER_SEED = L_LEVELS + LEAK_TICKS   # 38

SEEDS = [0, 1, 2, 3, 4]
DRY_RUN_SEEDS = [0]

# Pre-registered acceptance thresholds.
MONOTONE_TOL = 1e-9        # non-decreasing / non-increasing tolerance
RISE_MARGIN = 0.5          # V_s[p] peak - first must exceed this
LOCALITY_FLOOR = 1e-9      # fresh-region V_s must stay <= this
SPREAD_MARGIN = 0.5        # peak (V_s[p] - max_fresh) >= this (LOAD-BEARING)
LEAK_DECAY_FRAC = 0.05     # V_s[p] final <= frac * peak
# Readiness (substrate_not_ready_requeue) -- SAME statistic as C2 (range).
RANGE_FLOOR = 1e-6         # peak cross-region range must exceed this
VS_FLOOR = 1e-3            # peak V_s[p] magnitude readiness (secondary)


def _make_config(mode: str) -> StalenessAccumulatorConfig:
    return StalenessAccumulatorConfig(
        leak_factor=LEAK_FACTOR,
        attribution_mode=mode,
        staleness_clip=STALENESS_CLIP,
        drop_epsilon=DROP_EPSILON,
    )


def _build_regions(seed: int) -> Tuple[List[Anchor], int, List[Tuple[str, str]]]:
    """Build N_REGIONS real Anchors with disjoint stream mixtures; pick perturbed p.

    Returns (anchors, perturbed_index, region_keys) where region_keys[i] =
    (scale, segment_id) is the accumulator's per-region key for anchor i.
    """
    anchors: List[Anchor] = []
    region_keys: List[Tuple[str, str]] = []
    z = torch.zeros(4)
    for j in range(N_REGIONS):
        seg_id = "seg_s%d_r%d" % (seed, j)
        mixture = ("st_s%d_r%d_a" % (seed, j), "st_s%d_r%d_b" % (seed, j))
        anchors.append(Anchor(key=(SCALE, seg_id, mixture), z_world=z.clone()))
        region_keys.append((SCALE, seg_id))
    rng = random.Random(seed)
    p = rng.randrange(N_REGIONS)
    return anchors, p, region_keys


def _phase_a(seed: int, anchors: List[Anchor], p: int,
             region_keys: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Graded accumulation + region-locality (stream_overlap, all anchors active)."""
    acc = StalenessAccumulator(_make_config("stream_overlap"))
    mixture_p = anchors[p].key[2]
    key_p = region_keys[p]
    fresh_keys = [region_keys[j] for j in range(N_REGIONS) if j != p]

    vs_p_series: List[float] = []
    max_fresh_series: List[float] = []
    for i in range(L_LEVELS):
        bcast = BroadcastEvent(
            t=i,
            strength=DOSE,
            posterior=1.0,
            targets=[],
            source_scale=SCALE,
            source_segment_id_old=anchors[p].key[1],
            source_segment_id_new=anchors[p].key[1],
            source_sources=list(mixture_p),
        )
        # ALL anchors active -- fresh regions are active but non-overlapping.
        acc.integrate([bcast], anchors)
        snap = acc.snapshot()
        vs_p_series.append(float(snap.get(key_p, 0.0)))
        max_fresh_series.append(
            max((float(snap.get(k, 0.0)) for k in fresh_keys), default=0.0)
        )
        print("  [probe] seed=%d ep %d/%d phaseA vs_p=%.4f max_fresh=%.2e"
              % (seed, i + 1, STEPS_PER_SEED, vs_p_series[-1], max_fresh_series[-1]),
              flush=True)

    peak_vs_p = max(vs_p_series) if vs_p_series else 0.0
    peak_max_fresh = max(max_fresh_series) if max_fresh_series else 0.0
    return {
        "acc": acc,
        "key_p": key_p,
        "vs_p_series": vs_p_series,
        "max_fresh_series": max_fresh_series,
        "peak_vs_p": peak_vs_p,
        "peak_max_fresh": peak_max_fresh,
        "peak_range": peak_vs_p - peak_max_fresh,
    }


def _phase_b(seed: int, acc: StalenessAccumulator,
             key_p: Tuple[str, str], peak_vs_p: float) -> Dict[str, Any]:
    """Leak decay from the Phase-A peak."""
    leak_series: List[float] = [peak_vs_p]
    for i in range(LEAK_TICKS):
        acc.tick_leak()
        leak_series.append(float(acc.get(key_p)))
        print("  [probe] seed=%d ep %d/%d phaseB vs_p=%.3e"
              % (seed, L_LEVELS + i + 1, STEPS_PER_SEED, leak_series[-1]),
              flush=True)
    dropped = key_p not in acc.snapshot()
    return {
        "leak_series": leak_series,
        "leak_final": leak_series[-1],
        "dropped_from_snapshot": bool(dropped),
    }


def _phase_c(seed: int, anchors: List[Anchor], p: int,
             region_keys: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Equal-mode monotonicity control: active set = {perturbed anchor} only."""
    acc = StalenessAccumulator(_make_config("equal"))
    key_p = region_keys[p]
    fresh_keys = [region_keys[j] for j in range(N_REGIONS) if j != p]
    vs_p_series: List[float] = []
    for i in range(L_LEVELS):
        bcast = BroadcastEvent(
            t=i, strength=DOSE, posterior=1.0, targets=[], source_scale=SCALE,
            source_segment_id_old=anchors[p].key[1],
            source_segment_id_new=anchors[p].key[1],
            source_sources=list(anchors[p].key[2]),
        )
        acc.integrate([bcast], [anchors[p]])   # only perturbed anchor active
        vs_p_series.append(float(acc.snapshot().get(key_p, 0.0)))
    snap = acc.snapshot()
    fresh_absent = all(k not in snap for k in fresh_keys)
    non_decr = all(vs_p_series[i] >= vs_p_series[i - 1] - MONOTONE_TOL
                   for i in range(1, len(vs_p_series)))
    rises = (vs_p_series[-1] - vs_p_series[0]) >= RISE_MARGIN if vs_p_series else False
    return {
        "vs_p_series": vs_p_series,
        "fresh_absent": bool(fresh_absent),
        "equal_mode_monotone": bool(non_decr and rises and fresh_absent),
    }


def _score_seed(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    vs = a["vs_p_series"]
    # C1 -- monotone non-decreasing rise.
    c1_monotone = all(vs[i] >= vs[i - 1] - MONOTONE_TOL for i in range(1, len(vs)))
    c1_rise = (max(vs) - vs[0]) >= RISE_MARGIN if vs else False
    c1 = bool(c1_monotone and c1_rise)
    # C2 -- region-locality (LOAD-BEARING; routes on cross-region range/spread).
    c2_fresh_low = a["peak_max_fresh"] <= LOCALITY_FLOOR
    c2_spread = a["peak_range"] >= SPREAD_MARGIN
    c2 = bool(c2_fresh_low and c2_spread)
    # C3 -- leak decay.
    ls = b["leak_series"]
    c3_monotone = all(ls[i] <= ls[i - 1] + MONOTONE_TOL for i in range(1, len(ls)))
    c3_decayed = b["leak_final"] <= LEAK_DECAY_FRAC * (ls[0] if ls else 1.0)
    c3 = bool(c3_monotone and c3_decayed and b["dropped_from_snapshot"])
    return {
        "C1_monotone_rise": c1,
        "C1_monotone": bool(c1_monotone),
        "C1_rise": bool(c1_rise),
        "C2_region_locality": c2,
        "C2_fresh_low": bool(c2_fresh_low),
        "C2_spread": bool(c2_spread),
        "C3_leak_decay": c3,
        "C3_monotone": bool(c3_monotone),
        "C3_decayed": bool(c3_decayed),
        "C3_dropped": bool(b["dropped_from_snapshot"]),
        "seed_pass": bool(c1 and c2 and c3),
    }


def run_experiment(seeds: List[int]) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        print("Seed %d Condition staleness_probe" % seed, flush=True)
        anchors, p, region_keys = _build_regions(seed)
        a = _phase_a(seed, anchors, p, region_keys)
        b = _phase_b(seed, a["acc"], a["key_p"], a["peak_vs_p"])
        c = _phase_c(seed, anchors, p, region_keys)
        score = _score_seed(a, b)
        row = {
            "seed": seed,
            "perturbed_region": p,
            "peak_vs_p": a["peak_vs_p"],
            "peak_max_fresh": a["peak_max_fresh"],
            "peak_range": a["peak_range"],
            "vs_p_series": a["vs_p_series"],
            "max_fresh_series": a["max_fresh_series"],
            "leak_series": b["leak_series"],
            "leak_final": b["leak_final"],
            "dropped_from_snapshot": b["dropped_from_snapshot"],
            "equal_mode_monotone": c["equal_mode_monotone"],
            "equal_mode_vs_p_series": c["vs_p_series"],
        }
        row.update(score)
        per_seed.append(row)
        print("verdict: %s" % ("PASS" if score["seed_pass"] else "FAIL"), flush=True)

    # --- Readiness gate (substrate_not_ready_requeue) -- range-kind statistic. ---
    min_peak_range = min(r["peak_range"] for r in per_seed)
    min_peak_vs_p = min(r["peak_vs_p"] for r in per_seed)
    range_ready = min_peak_range > RANGE_FLOOR
    vs_ready = min_peak_vs_p > VS_FLOOR
    readiness_met = bool(range_ready and vs_ready)

    all_c1 = all(r["C1_monotone_rise"] for r in per_seed)
    all_c2 = all(r["C2_region_locality"] for r in per_seed)
    all_c3 = all(r["C3_leak_decay"] for r in per_seed)
    all_pass = all(r["seed_pass"] for r in per_seed)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "accumulator empty or flat across regions after real integrate(): "
            "min peak cross-region range %.3e <= RANGE_FLOOR %.3e (or peak V_s[p] "
            "%.3e <= VS_FLOOR %.3e); substrate did not populate -- requeue, not a "
            "weakens." % (min_peak_range, RANGE_FLOOR, min_peak_vs_p, VS_FLOOR)
        )
    elif all_pass:
        outcome = "PASS"
        label = "confirms_region_indexed_staleness_tracking"
        direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
    else:
        outcome = "FAIL"
        label = "staleness_accumulator_not_monotone_or_localized"
        direction = "weakens"
        non_degenerate = True
        degeneracy_reason = ""

    interpretation = {
        "label": label,
        "readiness_met": readiness_met,
        "all_seeds_C1_monotone_rise": all_c1,
        "all_seeds_C2_region_locality": all_c2,
        "all_seeds_C3_leak_decay": all_c3,
        "all_seeds_pass": all_pass,
        "min_peak_range": min_peak_range,
        "min_peak_vs_p": min_peak_vs_p,
        "preconditions": [
            {
                "name": "peak_cross_region_range_supra_floor",
                "description": ("perturbed region minus max fresh region at Phase-A "
                                "peak -- SAME statistic the load-bearing locality "
                                "criterion C2 routes on"),
                "measured": min_peak_range,
                "threshold": RANGE_FLOOR,
                "control": ("perturbed region after graded real integrate() vs fresh "
                            "non-overlapping active regions"),
                "met": bool(range_ready),
            },
            {
                "name": "peak_vs_p_level_supra_floor",
                "description": "perturbed-region V_s level at Phase-A peak (secondary)",
                "measured": min_peak_vs_p,
                "threshold": VS_FLOOR,
                "control": "perturbed region after L graded doses via integrate()",
                "met": bool(vs_ready),
            },
        ],
        "criteria_non_degenerate": {
            "C1_monotone_rise": bool(all_c1),
            "C2_region_locality": bool(all_c2),
            "C3_leak_decay": bool(all_c3),
        },
        "criteria": [
            {"name": "C2_region_locality_spread", "load_bearing": True,
             "passed": bool(all_c2)},
            {"name": "C1_monotone_rise", "load_bearing": False, "passed": bool(all_c1)},
            {"name": "C3_leak_decay", "load_bearing": False, "passed": bool(all_c3)},
        ],
    }

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "per_seed": per_seed,
        "interpretation": interpretation,
        "n_seeds": len(seeds),
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str,
                    seeds: List[int]) -> Dict[str, Any]:
    run_id = "%s_%s_v3" % (EXPERIMENT_TYPE, timestamp_utc)
    config = {
        "scale": SCALE,
        "n_regions": N_REGIONS,
        "l_levels": L_LEVELS,
        "dose": DOSE,
        "leak_ticks": LEAK_TICKS,
        "leak_factor": LEAK_FACTOR,
        "staleness_clip": STALENESS_CLIP,
        "drop_epsilon": DROP_EPSILON,
        "attribution_modes": ["stream_overlap", "equal"],
        "steps_per_seed": STEPS_PER_SEED,
        "seeds": list(seeds),
        "use_staleness_accumulator": True,
        "thresholds": {
            "monotone_tol": MONOTONE_TOL,
            "rise_margin": RISE_MARGIN,
            "locality_floor": LOCALITY_FLOOR,
            "spread_margin": SPREAD_MARGIN,
            "leak_decay_frac": LEAK_DECAY_FRAC,
            "range_floor": RANGE_FLOOR,
            "vs_floor": VS_FLOOR,
        },
    }
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "timestamp_utc": timestamp_utc,
        "interpretation": result["interpretation"],
        "per_seed": result["per_seed"],
        "n_seeds": result["n_seeds"],
        "config": config,
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "notes": (
            "Wall-independent action-free test of MECH-284. Drives the real "
            "StalenessAccumulator.integrate()/tick_leak() path with real "
            "BroadcastEvent + Anchor objects; DV = region-indexed snapshot() scalar. "
            "Readiness gate keys on peak cross-region RANGE (same statistic as the "
            "load-bearing locality criterion C2) -> substrate_not_ready_requeue / "
            "non_contributory if flat/empty (guards the V3-EXQ-688 vacuous-null; "
            "populated ONLY via real integrate(), not any default-1.0 path). "
            "GOV-REUSE-1: decisive region-indexed snapshot readout absent from all "
            "recorded manifests (455a/592f reference MECH-284 but never tested it) "
            "-> run. Re-derive brake: 0 substrate_ceiling/non_contributory autopsies "
            "tag MECH-284. No arm_results / no OFF training arm -> no baseline mint "
            "(pure-arithmetic substrate probe, nothing to re-train)."
        ),
    }
    return manifest


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke test (1 seed).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override output dir (default: REE_assembly evidence/experiments).")
    args = parser.parse_args()
    run_started = datetime.now(timezone.utc)

    seeds = list(DRY_RUN_SEEDS) if args.dry_run else list(SEEDS)
    result = run_experiment(seeds)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, seeds)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest["config"],
        seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - run_started).total_seconds(),
    )

    print("manifest: %s" % out_path, flush=True)
    print("outcome: %s label=%s direction=%s non_degenerate=%s "
          "readiness_met=%s all_pass=%s"
          % (result["outcome"], result["interpretation"]["label"],
             result["evidence_direction"], result["non_degenerate"],
             result["interpretation"]["readiness_met"],
             result["interpretation"]["all_seeds_pass"]),
          flush=True)

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    return outcome_emit, str(out_path), bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry)
