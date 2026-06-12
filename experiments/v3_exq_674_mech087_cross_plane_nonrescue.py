"""
V3-EXQ-674: MECH-087 cross-plane non-rescue (plane-level dissociation logic).

Tests the verbatim MECH-087 prediction (neuromodulatory_control_planes.md):
"selectively degrading the serotonin axis (map geometry) should produce
characteristic trajectory pathology that is NOT rescued by increasing dopamine
gain; conversely, degrading only the dopamine axis should produce a different
failure mode that IS rescued by dopamine augmentation, not by serotonin
adjustment."

This is the single V3-tractable experiment named in
REE_assembly/docs/architecture/receptor_subtype_intervention_layer.md Section 4
as licensing the receptor layer's plane-level dissociation logic. The
receptor-SUBTYPE distinctions themselves are out_of_domain for V3; only the
PLANE-level dissociation is V3-testable, and that is what this script measures.

SUBSTRATE MAPPING (operationalisation; the abstract->concrete rung):
  * DOPAMINE / trajectory-selection-gain plane (MECH-086) -> the E3 softmax
    temperature in E3TrajectorySelector.select() (e3_selector.py):
        probs = F.softmax(-scores / temperature)
    Low temperature = high selection gain (sharp, decisive selection =
    dopamine augmentation); high temperature = flattened selection
    (no trajectory wins = dopamine depletion / anhedonia-indecision).
    NOTE (load-bearing substrate detail): temperature only modulates the
    UNCOMMITTED (multinomial) branch -- the committed branch uses argmin(scores)
    regardless of temperature. So this probe forces the uncommitted regime
    (running_variance held above commitment_threshold) and verifies it in P0,
    so the dopamine knob is genuinely operative.
  * SEROTONIN / map-geometry-terrain plane (MECH-085) -> the per-candidate
    score landscape the selection runs over. A serotonin-terrain DEGRADATION
    distorts the landscape geometry so the score-ranking diverges from true
    trajectory quality (basins in the wrong place); a terrain ADJUSTMENT/RESCUE
    restores the correct geometry. The landscape is injected through the real
    E3 modulatory score_bias path over an otherwise-uniform base (all candidate
    trajectories share an identical z_world, so base scores cancel and the
    injected landscape IS the geometry the selection runs over).

DESIGN: a controlled probe of the REAL e3.select() softmax-selection stage --
no env, no training -- which isolates the two planes and their interaction
without the monostrategy / z_goal-collapse confounds that contaminate
ecological runs. The non-triviality is that the V3 DEFAULT selection stage must
exhibit the predicted hierarchical-ordering property (downstream gain cannot
compensate for upstream geometry); substrate features (modulatory authority,
stratified select, commit gate) could break the clean dissociation, and the
P0 readiness gate confirms the regime is the one in which the test is meaningful.

THE 2x2 (degrade serotonin-terrain vs degrade dopamine-selection)
        x (rescue-with-dopamine-gain vs rescue-with-terrain-adjustment),
plus 3 anchors (no-lesion control + the two lesion-only baselines):

  CTRL        : clean landscape, T_base          -> good (selects near-best)
  TERR_LESION : distorted landscape, T_base      -> bad  (terrain-lesion anchor)
  DOPA_LESION : clean landscape, T_high (flat)   -> bad  (dopamine-lesion anchor)
  A = TERR_LESION + DOPA_RESCUE  : distorted, T_low  -> predict NON-rescue
        (sharpening selection over a distorted landscape confidently commits to
         the wrong trajectory; MECH-086: "dopamine augmentation on a distorted
         serotonergic landscape produces rapid, confident commitment to
         distorted trajectories")
  B = TERR_LESION + TERRAIN_RESCUE : clean, T_base   -> predict RESCUE
        (restoring geometry restores the correct ranking)
  C = DOPA_LESION + DOPA_RESCUE   : clean, T_low      -> predict RESCUE
        (landscape was fine; sharpening restores decisive correct selection)
  D = DOPA_LESION + TERRAIN_RESCUE : clean, T_high    -> predict NON-rescue
        (landscape was already correct; "adjusting terrain" cannot fix flat
         selection -- the temperature lesion persists)

DISCRIMINATIVE METRIC: the TRUE (clean-landscape) quality of the E3-SELECTED
candidate -- i.e. the clean-landscape cost q at the index the real multinomial
selection actually picked (lower = better) -- plus selection decisiveness
(1 - normalised softmax entropy). The predicted signature is the axis-specific
rescue asymmetry: each lesion is rescued ONLY by its matching-axis intervention
(terrain lesion rescued by terrain-adjust NOT dopamine-gain; dopamine lesion
rescued by dopamine-gain NOT terrain-adjust).

NON-DEGENERACY / READINESS: _metrics.check_degeneracy over the per-cell quality
spread + a P0 readiness gate (positive controls) that self-routes to
substrate_not_ready_requeue if any lesion is inert: (1) the clean landscape is
genuinely discriminative, (2) the terrain distortion actually MOVES the argmin
(the same statistic the terrain-lesion-degrades-quality criterion routes on),
(3) temperature actually modulates selection decisiveness AND the regime is
uncommitted (the same statistic the dopamine knob routes on).

claim_ids = [MECH-087] ONLY. The temperature<->dopamine and landscape<->serotonin
mappings are interpretive operationalisation, not independent validations of
MECH-085 / MECH-086; per the claim_ids accuracy rule (err toward fewer tags),
only the cross-plane non-rescue prediction (MECH-087) is directly measured.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from _metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402
from ree_core.predictors.e2_fast import Trajectory  # noqa: E402
from ree_core.utils.config import E3Config  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_674_mech087_cross_plane_nonrescue"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-087"]

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44]
K = 8                       # candidate-pool size (typical CEM pool)
WORLD_DIM = 32              # E3Config default z_world dim
ACTION_DIM = 4             # default action space
N_PROBES = 30               # distinct landscapes per (seed, arm), shared across arms
N_DRAWS = 16                # real e3.select multinomial draws per probe

Q_SCALE = 4.0               # clean-landscape cost magnitude (max ~ Q_SCALE)
# Serotonin map-geometry lesion: blend the clean cost landscape toward an
# INDEPENDENT scrambled landscape (basins decoupled from true quality). TERR_MIX
# in [0,1]; 1.0 = fully value-uninformative geometry (the limiting case of
# MECH-085 map degradation -- the landscape no longer encodes true trajectory
# value, so no selection-gain setting can recover good trajectories from it).
TERR_MIX = 0.95

T_BASE = 1.0                # E3 default selection temperature
T_LOW = 0.05                # dopamine augmentation (sharp / high gain)
T_HIGH = 20.0               # dopamine depletion (flat / low gain)

# Forced uncommitted regime: hold running_variance above commitment_threshold so
# the multinomial (temperature-sensitive) branch is the one exercised.
FORCED_RUNNING_VARIANCE = 0.6   # > E3Config.commitment_threshold (0.40)

# Acceptance thresholds (pre-registered)
RESCUE_PASS = 0.5           # a matching-axis rescue must recover >= this fraction
NONRESCUE_MAX = 0.25        # a non-matching-axis rescue must recover <= this fraction
DISSOCIATION_MARGIN = 0.5   # rf_match - rf_mismatch must be >= this
SEED_MAJORITY = 2           # dissociation must hold on >= this many of 3 seeds

# P0 readiness floors (positive controls)
CLEAN_SPREAD_FLOOR = 1.0           # clean landscape best-vs-worst gap must exceed
TERRAIN_SHIFT_FLOOR = 0.5          # fraction of probes whose argmin the lesion moves
DOPA_ENTROPY_GAP_FLOOR = 0.4       # (norm. entropy @ T_high) - (@ T_low) must exceed
UNCOMMITTED_RATE_FLOOR = 0.99      # selection must be uncommitted in the probe regime

ARMS = ["CTRL", "TERR_LESION", "DOPA_LESION", "A", "B", "C", "D"]

# (uses_distorted_landscape, temperature) per arm
ARM_SPEC = {
    "CTRL":        (False, T_BASE),
    "TERR_LESION": (True,  T_BASE),
    "DOPA_LESION": (False, T_HIGH),
    "A":           (True,  T_LOW),   # terrain lesion + dopamine rescue
    "B":           (False, T_BASE),  # terrain lesion + terrain rescue (geometry restored)
    "C":           (False, T_LOW),   # dopamine lesion + dopamine rescue
    "D":           (False, T_HIGH),  # dopamine lesion + terrain rescue (no-op on temperature)
}


def _normalised_entropy(landscape: np.ndarray, temperature: float) -> float:
    """Normalised softmax(-landscape/T) entropy in [0,1]; 1 = flat selection."""
    logits = -landscape / max(temperature, 1e-9)
    logits = logits - logits.max()
    p = np.exp(logits)
    p = p / p.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / math.log(len(landscape)))


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


def _select_true_quality(e3, cands, landscape: np.ndarray, temperature: float):
    """Run ONE real e3.select() with the landscape injected via score_bias;
    return (true_clean_quality_unused_here, selected_index, committed)."""
    bias = torch.tensor(landscape, dtype=torch.float32)
    res = e3.select(cands, temperature=temperature, score_bias=bias)
    return int(res.selected_index), bool(res.committed)


def _run_cell(arm: str, seed: int, landscapes, e3, cands) -> Dict[str, Any]:
    uses_dist, temperature = ARM_SPEC[arm]
    sel_qualities: List[float] = []
    entropies: List[float] = []
    committed_flags: List[bool] = []
    for p, (q_clean, q_dist) in enumerate(landscapes):
        landscape = q_dist if uses_dist else q_clean
        entropies.append(_normalised_entropy(landscape, temperature))
        for _ in range(N_DRAWS):
            idx, committed = _select_true_quality(e3, cands, landscape, temperature)
            # TRUE quality is ALWAYS read off the CLEAN landscape (the oracle),
            # regardless of which landscape the selection ran over.
            sel_qualities.append(float(q_clean[idx]))
            committed_flags.append(committed)
        if (p + 1) % 10 == 0:
            print(f"  [probe] {arm} seed={seed} ep {p+1}/{N_PROBES}", flush=True)
    return {
        "arm": arm,
        "seed": seed,
        "uses_distorted_landscape": uses_dist,
        "temperature": temperature,
        "mean_true_quality": float(np.mean(sel_qualities)),
        "mean_selection_entropy": float(np.mean(entropies)),
        "uncommitted_rate": float(1.0 - np.mean(committed_flags)),
        "n_draws": len(sel_qualities),
    }


def _p0_readiness(seeds) -> tuple:
    """Positive-control measurements + the abort gate. Returns (preconditions,
    diag) on success; raises P0NotReady on a starved lesion."""
    clean_spreads: List[float] = []
    shift_flags: List[float] = []
    ent_low: List[float] = []
    ent_high: List[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(N_PROBES):
            q_clean, q_dist = _make_landscape(rng)
            clean_spreads.append(float(q_clean.max() - q_clean.min()))
            shift_flags.append(1.0 if int(np.argmin(q_dist)) != int(np.argmin(q_clean)) else 0.0)
            ent_low.append(_normalised_entropy(q_clean, T_LOW))
            ent_high.append(_normalised_entropy(q_clean, T_HIGH))

    # Confirm the probe regime is genuinely uncommitted via the REAL selector.
    e3 = _build_e3()
    cands = _build_candidates()
    rng = np.random.default_rng(seeds[0])
    committed = []
    for _ in range(20):
        q_clean, _ = _make_landscape(rng)
        _, c = _select_true_quality(e3, cands, q_clean, T_BASE)
        committed.append(c)
    uncommitted_rate = float(1.0 - np.mean(committed))

    diag = {
        "clean_landscape_spread_mean": float(np.mean(clean_spreads)),
        "terrain_argmin_shift_rate": float(np.mean(shift_flags)),
        "dopa_entropy_gap": float(np.mean(ent_high) - np.mean(ent_low)),
        "probe_uncommitted_rate": uncommitted_rate,
    }
    preconditions = p0_readiness_gate([
        {"name": "clean_landscape_discriminative",
         "measured": diag["clean_landscape_spread_mean"],
         "threshold": CLEAN_SPREAD_FLOOR, "direction": "lower"},
        {"name": "terrain_lesion_moves_argmin",
         "measured": diag["terrain_argmin_shift_rate"],
         "threshold": TERRAIN_SHIFT_FLOOR, "direction": "lower"},
        {"name": "dopamine_temperature_modulates_decisiveness",
         "measured": diag["dopa_entropy_gap"],
         "threshold": DOPA_ENTROPY_GAP_FLOOR, "direction": "lower"},
        {"name": "probe_regime_uncommitted",
         "measured": diag["probe_uncommitted_rate"],
         "threshold": UNCOMMITTED_RATE_FLOOR, "direction": "lower"},
    ])
    return preconditions, diag


def _rescue_fraction(q_lesion: float, q_cell: float, q_ctrl: float) -> float:
    denom = q_lesion - q_ctrl
    if abs(denom) < 1e-9:
        return 0.0
    return float((q_lesion - q_cell) / denom)


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global N_PROBES, N_DRAWS
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    seeds = SEEDS[:1] if dry_run else SEEDS
    # apply dry-run downscale via module globals the helpers read
    if dry_run:
        N_PROBES, N_DRAWS = 6, 4

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
            "degeneracy_reason": "P0 readiness unmet -- a lesion is inert (" + e.reason + ")",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
            "dry_run": dry_run,
        }
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
        out_path = out_dir / f"{run_id}.json"
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(manifest, fh, indent=2)
            print(f"Manifest written: {out_path}", flush=True)
        else:
            out_path = Path("/dev/null")
            print("Dry run -- manifest not written.", flush=True)
        print("Outcome: FAIL (substrate_not_ready_requeue)", flush=True)
        manifest["manifest_path"] = str(out_path)
        return manifest

    # ---- main measurement: 7 arms x N seeds (matched landscapes across arms) ----
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        # Generate the per-probe landscapes ONCE per seed; shared across all arms
        # so arms differ ONLY in lesion/rescue (matched-landscape control).
        lrng = np.random.default_rng(seed)
        landscapes = [_make_landscape(lrng) for _ in range(N_PROBES)]
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            cfg_slice = {
                "experiment_type": EXPERIMENT_TYPE, "arm": arm, "seed": seed,
                "K": K, "world_dim": WORLD_DIM, "action_dim": ACTION_DIM,
                "n_probes": N_PROBES, "n_draws": N_DRAWS,
                "q_scale": Q_SCALE, "terr_mix": TERR_MIX,
                "t_base": T_BASE, "t_low": T_LOW, "t_high": T_HIGH,
                "forced_running_variance": FORCED_RUNNING_VARIANCE,
                "uses_distorted_landscape": ARM_SPEC[arm][0],
                "temperature": ARM_SPEC[arm][1],
            }
            with arm_cell(seed, config_slice=cfg_slice,
                          script_path=Path(__file__)) as cell:
                e3 = _build_e3()
                cands = _build_candidates()
                row = _run_cell(arm, seed, landscapes, e3, cands)
                cell.stamp(row)
            rows.append(row)
            print("verdict: PASS", flush=True)

    # ---- aggregate ----
    def q(arm: str, seed=None) -> float:
        vals = [r["mean_true_quality"] for r in rows
                if r["arm"] == arm and (seed is None or r["seed"] == seed)]
        return float(np.mean(vals))

    q_ctrl = q("CTRL")
    q_terr = q("TERR_LESION")
    q_dopa = q("DOPA_LESION")

    rf_A = _rescue_fraction(q_terr, q("A"), q_ctrl)   # terrain lesion + dopamine rescue
    rf_B = _rescue_fraction(q_terr, q("B"), q_ctrl)   # terrain lesion + terrain rescue
    rf_C = _rescue_fraction(q_dopa, q("C"), q_ctrl)   # dopamine lesion + dopamine rescue
    rf_D = _rescue_fraction(q_dopa, q("D"), q_ctrl)   # dopamine lesion + terrain rescue

    # Per-seed dissociation (majority gate)
    terrain_seed_hits = 0
    dopa_seed_hits = 0
    per_seed = []
    for seed in seeds:
        qc, qt, qd = q("CTRL", seed), q("TERR_LESION", seed), q("DOPA_LESION", seed)
        a = _rescue_fraction(qt, q("A", seed), qc)
        b = _rescue_fraction(qt, q("B", seed), qc)
        c = _rescue_fraction(qd, q("C", seed), qc)
        d = _rescue_fraction(qd, q("D", seed), qc)
        terr_ok = (b - a >= DISSOCIATION_MARGIN) and (b >= RESCUE_PASS) and (a <= NONRESCUE_MAX)
        dopa_ok = (c - d >= DISSOCIATION_MARGIN) and (c >= RESCUE_PASS) and (d <= NONRESCUE_MAX)
        terrain_seed_hits += int(terr_ok)
        dopa_seed_hits += int(dopa_ok)
        per_seed.append({"seed": seed, "rf_A": a, "rf_B": b, "rf_C": c, "rf_D": d,
                         "terrain_dissociation": terr_ok, "dopamine_dissociation": dopa_ok})

    seed_majority = max(1, min(SEED_MAJORITY, len(seeds)))
    terrain_dissociation = terrain_seed_hits >= seed_majority
    dopamine_dissociation = dopa_seed_hits >= seed_majority
    passed = bool(terrain_dissociation and dopamine_dissociation)
    outcome = "PASS" if passed else "FAIL"

    # ---- non-degeneracy net (applies to evidence runs too) ----
    degen = check_degeneracy({
        "selected_true_quality_across_cells": [r["mean_true_quality"] for r in rows],
        "rescue_fraction_gap_per_seed": {
            "values": [s["rf_B"] - s["rf_A"] for s in per_seed]
                      + [s["rf_C"] - s["rf_D"] for s in per_seed]},
    })

    criteria = [
        {"name": "terrain_dissociation_rf_B_minus_rf_A", "load_bearing": True,
         "passed": terrain_dissociation},
        {"name": "dopamine_dissociation_rf_C_minus_rf_D", "load_bearing": True,
         "passed": dopamine_dissociation},
    ]
    # criteria_non_degenerate reflects whether the load-bearing metric had usable
    # spread (the dissociation criteria are vacuous if every cell scored alike).
    criteria_non_degenerate = {
        "terrain_dissociation_rf_B_minus_rf_A": bool(degen["non_degenerate"]),
        "dopamine_dissociation_rf_C_minus_rf_D": bool(degen["non_degenerate"]),
    }

    summary = {
        "q_ctrl": q_ctrl, "q_terrain_lesion": q_terr, "q_dopamine_lesion": q_dopa,
        "rescue_fraction_A_terrlesion_doparescue": rf_A,
        "rescue_fraction_B_terrlesion_terrrescue": rf_B,
        "rescue_fraction_C_dopalesion_doparescue": rf_C,
        "rescue_fraction_D_dopalesion_terrrescue": rf_D,
        "terrain_dissociation": terrain_dissociation,
        "dopamine_dissociation": dopamine_dissociation,
        "terrain_seed_hits": terrain_seed_hits,
        "dopamine_seed_hits": dopa_seed_hits,
        "seed_majority_required": seed_majority,
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
            "label": "cross_plane_non_rescue_confirmed" if passed
                     else "cross_plane_non_rescue_not_observed",
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
            "T_BASE": T_BASE, "T_LOW": T_LOW, "T_HIGH": T_HIGH,
            "RESCUE_PASS": RESCUE_PASS, "NONRESCUE_MAX": NONRESCUE_MAX,
            "DISSOCIATION_MARGIN": DISSOCIATION_MARGIN, "SEED_MAJORITY": SEED_MAJORITY,
        },
    }
    manifest.update(degen)  # non_degenerate / degeneracy_reason / degenerate_metrics

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = out_dir / f"{run_id}.json"
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    print(f"  rf_A(terr+dopa)={rf_A:.3f} rf_B(terr+terr)={rf_B:.3f} "
          f"rf_C(dopa+dopa)={rf_C:.3f} rf_D(dopa+terr)={rf_D:.3f}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    emit_outcome(
        outcome=str(result.get("outcome", "FAIL")),
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
