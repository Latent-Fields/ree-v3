"""
V3-EXQ-775: MECH-086 dopamine selection-gain plane DOSE-RESPONSE.

Tests MECH-086's OWN subject -- "dopamine as trajectory-selection GAIN plane"
(neuromodulatory_control_planes.md#mech-086) -- which no prior experiment has
directly measured. V3-EXQ-674 (cross_plane_non_rescue) exercised the SAME E3
softmax stage but was deliberately tagged claim_ids=[MECH-087] ONLY, explicitly
disclaiming that its incidental temperature manipulation "independently
validates MECH-086": 674 measured the ORDERING of two planes (MECH-087), not the
selection-gain plane's own dose-response. This experiment measures exactly that.

MECH-086 (verbatim claim content): dopamine "sets gain on trajectory selection
-- which rollout wins and how quickly -- by modulating attractor dynamics in the
already-valenced hippocampal system. ... dopamine optimises selection over
whatever landscape (serotonin) ... it receives -- cannot compensate for upstream
plane failures. Too little: indecision, flattened landscape, anhedonia. Too much:
premature commitment, aberrant salience."

Those four assertions decompose into three MEASURABLE, pre-registered signatures
of the selection-gain plane -- a monotone dose-response plus its two pathology
poles, and the downstream (cannot-manufacture-value) property:

  S1  GAIN SEMANTICS (which/how-quickly wins): as dopamine gain rises across a
      temperature ladder, the REAL selector's realised decisiveness increases
      MONOTONICALLY, on BOTH a clean and a value-decoupled landscape (gain
      sharpens selection regardless of what the landscape encodes).
  S2  DOWNSTREAM / VALUE-CONDITIONAL BENEFIT (cannot compensate upstream): on a
      CLEAN landscape higher gain IMPROVES selected true quality (sharper
      exploitation of a correct ranking); on a DISTORTED (value-decoupled)
      landscape higher gain does NOT improve quality -- dopamine optimises
      selection over whatever landscape it receives and cannot manufacture value
      the landscape does not encode. The benefit-of-gain is landscape-CONDITIONAL.
  S3  TWO PATHOLOGY POLES:
        - INDECISION / anhedonia (too little gain): at the lowest gain the
          selection is near-uniform (decisiveness ~ 0) on both landscapes.
        - ABERRANT SALIENCE (too much gain over a distorted landscape): at the
          highest gain on the distorted landscape the selection is CONFIDENT
          (high decisiveness) while true quality is BAD -- confidence decoupled
          from correctness = confident commitment to the wrong trajectory.

SUBSTRATE MAPPING (interpretive operationalisation; the abstract->concrete rung,
identical to 674 -- receptor-subtype distinctions are out_of_domain for V3, only
the plane-level selection-gain semantics are V3-testable):
  * DOPAMINE / selection-gain plane (MECH-086) -> the E3 softmax temperature in
    E3TrajectorySelector.select(): probs = softmax(-scores / temperature). Low
    temperature = high selection gain (sharp, decisive = dopamine augmentation);
    high temperature = flattened selection (dopamine depletion / anhedonia). The
    temperature knob only modulates the UNCOMMITTED (multinomial) branch -- the
    committed branch is a temperature-blind argmin -- so this probe forces and
    P0-verifies the uncommitted regime, exactly as 674.
  * The per-candidate score landscape the selection runs over is the geometry the
    gain plane acts on. A CLEAN landscape encodes true trajectory value; a
    DISTORTED landscape (blended 0.95 toward an independent scramble) decouples
    the score ranking from true value -- the limiting case of upstream (MECH-085)
    map degradation. The landscape is injected through the real E3 score_bias
    path over an identical-z_world candidate pool (base scores cancel, so the
    injected landscape IS the geometry selection runs over).

DESIGN: a controlled probe of the REAL e3.select() softmax-selection stage -- no
env, no training -- over a GAIN-LADDER x LANDSCAPE factorial (5 temperatures x
{clean, distorted}). The non-triviality is that the V3 DEFAULT selection stage
must exhibit the selection-gain dose-response MECH-086 asserts: substrate
features (modulatory authority, eligibility envelope, commit gate, stratified
select) could break the clean monotone dose-response or the landscape-conditional
benefit, and the P0 readiness gate (positive controls) confirms the regime is the
one in which the test is meaningful.

DECISIVENESS is measured EMPIRICALLY from the real selector's draws (per-probe
normalised entropy of the actual e3.select() multinomial picks over the K
candidates, then averaged over probes), NOT from the analytic softmax of the
landscape -- so S1/S3 are genuine measurements of the substrate's realised
behaviour, not a mathematical identity. TRUE QUALITY is the clean-landscape cost
at the index the real selection actually picked (lower = better), always read off
the CLEAN landscape (the oracle) regardless of which landscape selection ran over.

claim_ids = [MECH-086] ONLY. The temperature<->dopamine-gain and
landscape<->serotonin-map mappings are interpretive operationalisation; per the
claim_ids accuracy rule this experiment's PASS/FAIL bears directly and ONLY on
MECH-086's asserted selection-gain semantics (674 owns the MECH-087
cross-plane-ordering result).
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from _metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.utils.config import E3Config  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_775_mech086_selection_gain_dose_response"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-086"]

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44]
K = 8                       # candidate-pool size (typical CEM pool)
WORLD_DIM = 32              # E3Config default z_world dim
ACTION_DIM = 4             # default action space
N_PROBES = 30               # distinct landscapes per (seed, arm), shared across arms
N_DRAWS = 40                # real e3.select multinomial draws per probe (>=40 keeps
                            # the finite-sample entropy bias small at K=8 so the
                            # low-gain decisiveness is a faithful near-uniform read)

Q_SCALE = 4.0               # clean-landscape cost magnitude (max ~ Q_SCALE)
# Serotonin map-geometry lesion: blend the clean cost landscape toward an
# INDEPENDENT scrambled landscape (basins decoupled from true quality). TERR_MIX
# in [0,1]; 0.95 = value-uninformative geometry (the limiting case of MECH-085
# map degradation -- no selection-gain setting can recover good trajectories).
TERR_MIX = 0.95

# Dopamine-gain ladder (E3 selection temperature). gain = 1/T, ascending:
# g0 (lowest gain / most flattened) -> g4 (highest gain / sharpest). g2 == the
# E3 default temperature (T=1.0). Endpoints match 674 (T_HIGH=20, T_LOW=0.05).
GAIN_TEMPERATURES = [20.0, 4.0, 1.0, 0.2, 0.05]
N_GAINS = len(GAIN_TEMPERATURES)
LANDSCAPES = ["clean", "dist"]

# Forced uncommitted regime: hold running_variance above commitment_threshold so
# the multinomial (temperature-sensitive) branch is the one exercised.
FORCED_RUNNING_VARIANCE = 0.6   # > E3Config.commitment_threshold (0.40)

# ---- Acceptance thresholds (pre-registered; derived from the landscape
# construction q ~ uniform(0, Q_SCALE), K=8: E[min]~0.44, E[mean]~2.0, so the
# clean best-vs-chance quality gap ~1.56; NOT from the run's own statistics) ----
# S1 gain monotonicity: the REAL selector's realised decisiveness rises with gain
#     across the ladder, on both landscapes. Tested as a Spearman rank correlation
#     between gain-index and decisiveness (>= this floor). A single adjacent near-
#     tie swap at the near-uniform low-gain end drops 5-point Spearman to exactly
#     0.9, so 0.9 tolerates that measurement noise while still demanding an
#     overwhelmingly monotone trend (a genuine substrate test: the trend breaks if
#     the commit gate or eligibility pruning stops temperature governing selection).
S1_SPEARMAN_MIN = 0.9
# S2 downstream / value-conditional benefit: quality-improvement with gain
#     (Q@g0 - Q@g4; positive = higher gain gives better/lower cost).
BENEFIT_MARGIN = 1.0        # clean:    improvement must be >= this
NONBENEFIT_MAX = 0.5        # distorted: improvement must be <= this (no rescue)
# S3a indecision pole: decisiveness at the lowest gain must be near-uniform.
INDECISION_DECISIVENESS_MAX = 0.35
# S3b aberrant-salience pole: confident (high decisiveness) at highest gain over
#     the DISTORTED landscape, while true quality is bad (decoupled from clean).
ABERRANT_DECISIVENESS_MIN = 0.75
ABERRANT_QUALITY_GAP = 0.75  # (Q_dist@g4 - Q_clean@g4) must be >= this
SEED_MAJORITY = 2           # each signature must hold on >= this many of 3 seeds

# ---- P0 readiness floors (positive controls) ----
CLEAN_SPREAD_FLOOR = 1.0            # clean landscape best-vs-worst gap must exceed
TERRAIN_SHIFT_FLOOR = 0.5           # fraction of probes whose argmin the lesion moves
DECISIVENESS_GAP_FLOOR = 0.4        # empirical decisiveness(T_low) - decisiveness(T_high)
UNCOMMITTED_RATE_FLOOR = 0.99       # selection must be uncommitted in the probe regime

# (gain_index, landscape_kind) per arm; 5 gains x 2 landscapes = 10 arms.
ARMS: List[str] = [f"g{gi}_{kind}" for gi in range(N_GAINS) for kind in LANDSCAPES]


def _arm_spec(arm: str) -> Tuple[int, str, float, bool]:
    """arm 'gI_kind' -> (gain_index, kind, temperature, uses_distorted)."""
    gi_str, kind = arm.split("_")
    gi = int(gi_str[1:])
    return gi, kind, GAIN_TEMPERATURES[gi], (kind == "dist")


def _spearman_vs_gain(values: List[float]) -> float:
    """Spearman rank correlation between the gain-index order [0..n-1] and
    `values` (decisiveness in gain order). +1 = perfectly monotone increasing.
    Average-rank tie handling; no scipy dependency."""
    n = len(values)
    if n < 2:
        return 0.0
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    x = list(range(n))
    mx = sum(x) / n
    my = sum(ranks) / n
    num = sum((x[i] - mx) * (ranks[i] - my) for i in range(n))
    den = (sum((x[i] - mx) ** 2 for i in range(n)) ** 0.5) * \
          (sum((ranks[i] - my) ** 2 for i in range(n)) ** 0.5)
    return float(num / den) if den > 0.0 else 0.0


def _empirical_decisiveness(selected_indices: List[int]) -> float:
    """1 - normalised entropy of the EMPIRICAL selection distribution over the K
    candidates for ONE probe (16-24 real e3.select draws over one landscape).
    1 = fully decisive (all draws on one candidate); 0 = uniform (indecision)."""
    counts = np.bincount(selected_indices, minlength=K).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0.0]
    ent = -np.sum(nz * np.log(nz))
    return float(1.0 - ent / math.log(K))


def _make_landscape(rng: np.random.Generator):
    """A clean cost landscape q (lower = better) with a clear best, plus its
    geometry-degraded counterpart (the serotonin-terrain lesion): a blend toward
    an INDEPENDENT scrambled landscape so the distorted geometry's basins are
    decoupled from true trajectory value."""
    q_clean = rng.uniform(0.0, Q_SCALE, size=K).astype(np.float64)
    scramble = rng.uniform(0.0, Q_SCALE, size=K).astype(np.float64)
    q_dist = (1.0 - TERR_MIX) * q_clean + TERR_MIX * scramble
    return q_clean, q_dist


def _build_candidates() -> List[Trajectory]:
    """K candidate trajectories with an IDENTICAL z_world (so the real E3
    score_trajectory base is uniform across candidates and the injected
    score_bias landscape IS the geometry the selection runs over) and distinct
    one-hot first actions."""
    z_world = torch.zeros(1, WORLD_DIM)
    z_self = torch.zeros(1, WORLD_DIM)
    cands: List[Trajectory] = []
    for k in range(K):
        a = torch.zeros(1, 1, ACTION_DIM)
        a[0, 0, k % ACTION_DIM] = 1.0
        cands.append(Trajectory(
            states=[z_self.clone(), z_self.clone()],
            actions=a,
            world_states=[z_world.clone(), z_world.clone()],
        ))
    return cands


def _build_e3() -> E3TrajectorySelector:
    e3 = E3TrajectorySelector(E3Config(world_dim=WORLD_DIM))
    # Force the uncommitted regime so softmax temperature is the operative
    # selection-gain knob (the committed branch is a temperature-blind argmin).
    e3._running_variance = FORCED_RUNNING_VARIANCE
    return e3


def _select_once(e3, cands, landscape: np.ndarray, temperature: float):
    """Run ONE real e3.select() with the landscape injected via score_bias;
    return (selected_index, committed)."""
    bias = torch.tensor(landscape, dtype=torch.float32)
    res = e3.select(cands, temperature=temperature, score_bias=bias)
    return int(res.selected_index), bool(res.committed)


def _run_cell(arm: str, seed: int, landscapes, e3, cands) -> Dict[str, Any]:
    gi, kind, temperature, uses_dist = _arm_spec(arm)
    per_probe_decisiveness: List[float] = []
    sel_true_qualities: List[float] = []
    committed_flags: List[bool] = []
    for p, (q_clean, q_dist) in enumerate(landscapes):
        landscape = q_dist if uses_dist else q_clean
        probe_idx: List[int] = []
        for _ in range(N_DRAWS):
            idx, committed = _select_once(e3, cands, landscape, temperature)
            probe_idx.append(idx)
            # TRUE quality is ALWAYS read off the CLEAN landscape (the oracle),
            # regardless of which landscape the selection ran over.
            sel_true_qualities.append(float(q_clean[idx]))
            committed_flags.append(committed)
        per_probe_decisiveness.append(_empirical_decisiveness(probe_idx))
        if (p + 1) % 10 == 0 or (p + 1) == N_PROBES:
            print(f"  [probe] {arm} seed={seed} ep {p+1}/{N_PROBES}", flush=True)
    return {
        "arm": arm,
        "seed": seed,
        "gain_index": gi,
        "landscape": kind,
        "temperature": temperature,
        "uses_distorted_landscape": uses_dist,
        "mean_true_quality": float(np.mean(sel_true_qualities)),
        "mean_decisiveness": float(np.mean(per_probe_decisiveness)),
        "uncommitted_rate": float(1.0 - np.mean(committed_flags)),
        "n_draws": len(sel_true_qualities),
    }


def _p0_readiness(seeds) -> tuple:
    """Positive-control measurements + the abort gate. Returns (preconditions,
    diag) on success; raises P0NotReady on a starved control.

    The 'temperature modulates decisiveness' readiness asserts the SAME
    empirical-decisiveness statistic S1/S3 route on (per the same-statistic rule),
    measured on the positive-control clean landscape at the ladder endpoints.
    """
    clean_spreads: List[float] = []
    shift_flags: List[float] = []

    e3 = _build_e3()
    cands = _build_candidates()
    dec_low: List[float] = []    # empirical decisiveness at T_low  (highest gain)
    dec_high: List[float] = []   # empirical decisiveness at T_high (lowest gain)
    committed_probe: List[bool] = []
    t_high = GAIN_TEMPERATURES[0]
    t_low = GAIN_TEMPERATURES[-1]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(N_PROBES):
            q_clean, q_dist = _make_landscape(rng)
            clean_spreads.append(float(q_clean.max() - q_clean.min()))
            shift_flags.append(1.0 if int(np.argmin(q_dist)) != int(np.argmin(q_clean)) else 0.0)
            idx_low: List[int] = []
            idx_high: List[int] = []
            for _ in range(N_DRAWS):
                i_l, c_l = _select_once(e3, cands, q_clean, t_low)
                i_h, _ = _select_once(e3, cands, q_clean, t_high)
                idx_low.append(i_l)
                idx_high.append(i_h)
                committed_probe.append(c_l)
            dec_low.append(_empirical_decisiveness(idx_low))
            dec_high.append(_empirical_decisiveness(idx_high))

    diag = {
        "clean_landscape_spread_mean": float(np.mean(clean_spreads)),
        "terrain_argmin_shift_rate": float(np.mean(shift_flags)),
        "decisiveness_gap_low_minus_high": float(np.mean(dec_low) - np.mean(dec_high)),
        "probe_uncommitted_rate": float(1.0 - np.mean(committed_probe)),
    }
    preconditions = p0_readiness_gate([
        {"name": "clean_landscape_discriminative",
         "measured": diag["clean_landscape_spread_mean"],
         "threshold": CLEAN_SPREAD_FLOOR, "direction": "lower",
         "control": "clean landscape best-vs-worst cost gap"},
        {"name": "distorted_landscape_decouples_value",
         "measured": diag["terrain_argmin_shift_rate"],
         "threshold": TERRAIN_SHIFT_FLOOR, "direction": "lower",
         "control": "fraction of probes whose argmin the lesion moves"},
        {"name": "gain_modulates_decisiveness",
         "measured": diag["decisiveness_gap_low_minus_high"],
         "threshold": DECISIVENESS_GAP_FLOOR, "direction": "lower",
         "control": "empirical decisiveness gap high-gain vs low-gain on clean control "
                    "(SAME statistic S1/S3 route on)"},
        {"name": "probe_regime_uncommitted",
         "measured": diag["probe_uncommitted_rate"],
         "threshold": UNCOMMITTED_RATE_FLOOR, "direction": "lower",
         "control": "real selector commit rate in the forced-uncommitted probe regime"},
    ])
    return preconditions, diag


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global N_PROBES, N_DRAWS
    import time
    t0 = time.perf_counter()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        # keep N_DRAWS realistic so the low-gain decisiveness is not small-sample
        # inflated; shrink only the probe count for a fast smoke.
        N_PROBES, N_DRAWS = 6, 40

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

    # ---- P0 readiness (positive controls + abort gate) ----
    try:
        preconditions, p0_diag = _p0_readiness(seeds)
    except P0NotReady as e:
        manifest = {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": "diagnostic",
            "outcome": "FAIL",
            "timestamp_utc": timestamp,
            "non_degenerate": False,
            "degeneracy_reason": "P0 readiness unmet -- a positive control is inert (" + e.reason + ")",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
            "dry_run": dry_run,
        }
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = write_flat_manifest(
                manifest, out_dir, dry_run=False,
                config={"experiment_type": EXPERIMENT_TYPE, "seeds": SEEDS, "K": K,
                        "gain_temperatures": GAIN_TEMPERATURES, "terr_mix": TERR_MIX},
                seeds=SEEDS, script_path=Path(__file__), started_at=t0,
            )
            print(f"Manifest written: {out_path}", flush=True)
        else:
            out_path = Path("/dev/null")
            print("Dry run -- manifest not written.", flush=True)
        print("Outcome: FAIL (substrate_not_ready_requeue)", flush=True)
        manifest["manifest_path"] = str(out_path)
        return manifest

    # ---- main measurement: 10 arms x N seeds (matched landscapes across arms) ----
    rows: List[Dict[str, Any]] = []
    full_config = {
        "experiment_type": EXPERIMENT_TYPE, "K": K, "world_dim": WORLD_DIM,
        "action_dim": ACTION_DIM, "n_probes": N_PROBES, "n_draws": N_DRAWS,
        "q_scale": Q_SCALE, "terr_mix": TERR_MIX,
        "gain_temperatures": GAIN_TEMPERATURES,
        "forced_running_variance": FORCED_RUNNING_VARIANCE, "seeds": SEEDS,
    }
    for seed in seeds:
        # Generate the per-probe landscapes ONCE per seed; shared across all arms
        # so arms differ ONLY in gain/landscape (matched-landscape control).
        lrng = np.random.default_rng(seed)
        landscapes = [_make_landscape(lrng) for _ in range(N_PROBES)]
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            gi, kind, temperature, uses_dist = _arm_spec(arm)
            cfg_slice = dict(full_config)
            cfg_slice.update({"arm": arm, "seed": seed, "gain_index": gi,
                              "landscape": kind, "temperature": temperature,
                              "uses_distorted_landscape": uses_dist})
            with arm_cell(seed, config_slice=cfg_slice,
                          script_path=Path(__file__)) as cell:
                e3 = _build_e3()
                cands = _build_candidates()
                row = _run_cell(arm, seed, landscapes, e3, cands)
                cell.stamp(row)
            rows.append(row)
            print("verdict: PASS", flush=True)

    # ---- aggregate: decisiveness[kind][gain], quality[kind][gain] per seed ----
    def _cell(seed, gi, kind, field):
        for r in rows:
            if r["seed"] == seed and r["gain_index"] == gi and r["landscape"] == kind:
                return r[field]
        raise KeyError((seed, gi, kind, field))

    def _mean_over_seeds(gi, kind, field):
        return float(np.mean([_cell(s, gi, kind, field) for s in seeds]))

    # Oracle references (context only, not load-bearing): chance = mean cost,
    # best = min cost, averaged over the shared landscapes.
    oracle_chance, oracle_best = [], []
    for seed in seeds:
        lrng = np.random.default_rng(seed)
        ls = [_make_landscape(lrng)[0] for _ in range(N_PROBES)]
        oracle_chance.append(float(np.mean([q.mean() for q in ls])))
        oracle_best.append(float(np.mean([q.min() for q in ls])))

    per_seed: List[Dict[str, Any]] = []
    s1_hits = s2_hits = s3a_hits = s3b_hits = 0
    for seed in seeds:
        dec_clean = [_cell(seed, gi, "clean", "mean_decisiveness") for gi in range(N_GAINS)]
        dec_dist = [_cell(seed, gi, "dist", "mean_decisiveness") for gi in range(N_GAINS)]
        q_clean = [_cell(seed, gi, "clean", "mean_true_quality") for gi in range(N_GAINS)]
        q_dist = [_cell(seed, gi, "dist", "mean_true_quality") for gi in range(N_GAINS)]

        # S1: decisiveness rises monotonically with gain (Spearman >= floor) on
        #     BOTH landscapes -- gain governs selection sharpness regardless of
        #     what the landscape encodes.
        rho_clean = _spearman_vs_gain(dec_clean)
        rho_dist = _spearman_vs_gain(dec_dist)
        s1 = bool(rho_clean >= S1_SPEARMAN_MIN and rho_dist >= S1_SPEARMAN_MIN)

        # S2: value-conditional benefit-of-gain (Q@g0 - Q@g4; positive = improvement).
        benefit_clean = q_clean[0] - q_clean[N_GAINS - 1]
        benefit_dist = q_dist[0] - q_dist[N_GAINS - 1]
        s2 = bool(benefit_clean >= BENEFIT_MARGIN and benefit_dist <= NONBENEFIT_MAX)

        # S3a: indecision pole -- lowest gain near-uniform on both landscapes.
        s3a = bool(dec_clean[0] <= INDECISION_DECISIVENESS_MAX
                   and dec_dist[0] <= INDECISION_DECISIVENESS_MAX)

        # S3b: aberrant salience -- confident at highest gain over distorted while
        #      true quality is decoupled (worse than clean-landscape high gain).
        aberrant_quality_gap = q_dist[N_GAINS - 1] - q_clean[N_GAINS - 1]
        s3b = bool(dec_dist[N_GAINS - 1] >= ABERRANT_DECISIVENESS_MIN
                   and aberrant_quality_gap >= ABERRANT_QUALITY_GAP)

        s1_hits += int(s1); s2_hits += int(s2); s3a_hits += int(s3a); s3b_hits += int(s3b)
        per_seed.append({
            "seed": seed,
            "decisiveness_clean": dec_clean, "decisiveness_dist": dec_dist,
            "quality_clean": q_clean, "quality_dist": q_dist,
            "benefit_of_gain_clean": float(benefit_clean),
            "benefit_of_gain_dist": float(benefit_dist),
            "aberrant_quality_gap": float(aberrant_quality_gap),
            "spearman_gain_decisiveness_clean": float(rho_clean),
            "spearman_gain_decisiveness_dist": float(rho_dist),
            "S1_gain_monotonicity": s1,
            "S2_value_conditional_benefit": s2,
            "S3a_indecision_pole": s3a,
            "S3b_aberrant_salience_pole": s3b,
        })

    seed_majority = max(1, min(SEED_MAJORITY, len(seeds)))
    S1 = s1_hits >= seed_majority
    S2 = s2_hits >= seed_majority
    S3a = s3a_hits >= seed_majority
    S3b = s3b_hits >= seed_majority
    passed = bool(S1 and S2 and S3a and S3b)
    outcome = "PASS" if passed else "FAIL"

    # ---- non-degeneracy net (applies to evidence runs too) ----
    degen = check_degeneracy({
        "selected_true_quality_across_cells": [r["mean_true_quality"] for r in rows],
        "decisiveness_across_cells": [r["mean_decisiveness"] for r in rows],
    })

    criteria = [
        {"name": "S1_gain_monotonicity", "load_bearing": True, "passed": S1},
        {"name": "S2_value_conditional_benefit", "load_bearing": True, "passed": S2},
        {"name": "S3a_indecision_pole", "load_bearing": True, "passed": S3a},
        {"name": "S3b_aberrant_salience_pole", "load_bearing": True, "passed": S3b},
    ]
    criteria_non_degenerate = {
        c["name"]: bool(degen["non_degenerate"]) for c in criteria
    }

    summary = {
        "S1_gain_monotonicity": S1, "S2_value_conditional_benefit": S2,
        "S3a_indecision_pole": S3a, "S3b_aberrant_salience_pole": S3b,
        "s1_seed_hits": s1_hits, "s2_seed_hits": s2_hits,
        "s3a_seed_hits": s3a_hits, "s3b_seed_hits": s3b_hits,
        "seed_majority_required": seed_majority,
        "decisiveness_clean_by_gain": [_mean_over_seeds(gi, "clean", "mean_decisiveness") for gi in range(N_GAINS)],
        "decisiveness_dist_by_gain": [_mean_over_seeds(gi, "dist", "mean_decisiveness") for gi in range(N_GAINS)],
        "quality_clean_by_gain": [_mean_over_seeds(gi, "clean", "mean_true_quality") for gi in range(N_GAINS)],
        "quality_dist_by_gain": [_mean_over_seeds(gi, "dist", "mean_true_quality") for gi in range(N_GAINS)],
        "oracle_chance_cost": float(np.mean(oracle_chance)),
        "oracle_best_cost": float(np.mean(oracle_best)),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if passed else "weakens",
        "timestamp_utc": timestamp,
        "dry_run": dry_run,
        "p0_readiness": p0_diag,
        "interpretation": {
            "label": "selection_gain_dose_response_confirmed" if passed
                     else "selection_gain_dose_response_not_observed",
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "acceptance_criteria": summary,
        "summary": summary,
        "per_seed": per_seed,
        "arm_results": rows,
        "constants": {
            "SEEDS": SEEDS, "K": K, "N_PROBES": N_PROBES, "N_DRAWS": N_DRAWS,
            "Q_SCALE": Q_SCALE, "TERR_MIX": TERR_MIX,
            "GAIN_TEMPERATURES": GAIN_TEMPERATURES,
            "S1_SPEARMAN_MIN": S1_SPEARMAN_MIN, "BENEFIT_MARGIN": BENEFIT_MARGIN,
            "NONBENEFIT_MAX": NONBENEFIT_MAX,
            "INDECISION_DECISIVENESS_MAX": INDECISION_DECISIVENESS_MAX,
            "ABERRANT_DECISIVENESS_MIN": ABERRANT_DECISIVENESS_MIN,
            "ABERRANT_QUALITY_GAP": ABERRANT_QUALITY_GAP,
            "SEED_MAJORITY": SEED_MAJORITY,
        },
    }
    manifest.update(degen)  # non_degenerate / degeneracy_reason / degenerate_metrics

    # arm_results already in the manifest; write_flat_manifest stamps the always-
    # core (hoisting substrate_hash from the per-cell arm_fingerprints) and
    # enforces the run_id/_v3 + outcome identity invariants.
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = write_flat_manifest(
            manifest, out_dir, dry_run=False, config=full_config,
            seeds=SEEDS, script_path=Path(__file__), started_at=t0,
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    print(f"  S1(mono)={S1} S2(benefit)={S2} S3a(indecision)={S3a} S3b(aberrant)={S3b}", flush=True)
    print(f"  dec_clean_by_gain={['%.2f' % v for v in summary['decisiveness_clean_by_gain']]}", flush=True)
    print(f"  dec_dist_by_gain ={['%.2f' % v for v in summary['decisiveness_dist_by_gain']]}", flush=True)
    print(f"  q_clean_by_gain  ={['%.2f' % v for v in summary['quality_clean_by_gain']]}", flush=True)
    print(f"  q_dist_by_gain   ={['%.2f' % v for v in summary['quality_dist_by_gain']]}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(result.get("manifest_path", "/dev/null")),
        dry_run=args.dry_run,
    )
