"""V3-EXQ-869a: MECH-267 mode-conditioned hippocampal proposals, content-
persistence retest with BOTH mechanisms active (EVIDENCE).

supersedes: V3-EXQ-869

WHY THIS IS A LETTERED RETEST, NOT A NEW EXQ (CLAUDE.md EXQ versioning
policy): the scientific question is IDENTICAL to V3-EXQ-869 -- does
MECH-267's mode-conditioned proposal content survive HippocampalModule's
own multi-iteration CEM elite-refit at the production
num_cem_iterations=3 setting? V3-EXQ-869 (2026-08-02, 30 seeds,
pre-registered, USER-CONFIRMED failure_autopsy_V3-EXQ-869_2026-08-02.json)
answered FAIL for the mechanism as it was implemented at the time
(noise-scale modulation only): C0 (iters=1 manipulation check) PASSED
cleanly (gaps 0.031-0.092 vs 0.015 floor) but C1 (iters=3, production
default) FAILED (gaps -0.004 to +0.004 vs 0.01 floor, 0/30 seeds showed
the predicted ordering). The autopsy's determination was
competence_implementation_gap, not claim falsification: MECH-267's own
2026-04-20 registration and the original 2026-04-27 lit-pull name TWO
mechanisms (noise-scale + horizon-depth), and only the first was built.
The second -- SD-MECH267-HORIZON-DEPTH, mode-conditioned CEM
elite-selection scoring-window depth (Wikenheiser & Redish 2015) -- was
subsequently built (ree-v3 e0117eea8b, 2026-08-02; design doc
REE_assembly/docs/architecture/sd_mech267_horizon_depth_modulation.md),
completing the claim's own two-mechanism text. This script is the
follow-up retest that V3-EXQ-869's own manifest and claims.yaml
implementation_note explicitly call for (V3-EXQ-869 manifest fields
pending_retest_after_substrate:true /
superseded_by_substrate:SD-MECH267-HORIZON-DEPTH@2026-08-02).

WHAT CHANGED VS V3-EXQ-869, AND WHAT DID NOT: the experimental design,
DV, pairwise-gap thresholds (FLOOR_DIAGNOSTIC/FLOOR_PRODUCTION), modes,
seeds, and HippocampalModule construction are UNCHANGED from V3-EXQ-869
-- this is a same-question re-queue, not a redesign. The ONLY substantive
difference is that HippocampalConfig.mode_horizon_scale now has live,
non-empty defaults (external_task=0.5, internal_planning=1.0,
internal_replay=0.7, offline_consolidation=1.0), gated by the SAME
mode_conditioning_enabled flag as mode_noise_scale (see config.py, SD-
MECH267-HORIZON-DEPTH block). Because `_make_hippocampal` here (as in
869) constructs HippocampalConfig with mode_conditioning_enabled=True and
leaves mode_horizon_scale unset (so it takes the live config default),
BOTH mechanisms are active simultaneously in every cell of this run --
exactly the "both mechanisms active" condition the substrate build's own
implementation_note says was never tested. This is confirmed at runtime,
not merely asserted: see MECHANISM ACTIVATION CHECK below.

MECHANISM ACTIVATION CHECK (new in 869a, absent from 869 since horizon-
depth did not exist yet): every cell records
`mode_noise_scale_used` (hip._last_mode_noise_scale) AND
`mode_horizon_scale_used` / `effective_horizon_used`
(hip._last_mode_horizon_scale / hip._last_effective_horizon) so the
manifest itself proves both mechanisms fired, rather than relying on
config inspection. With HORIZON=4 (unchanged from 869) and the config
defaults above, the four modes' effective_horizon values are
external_task=2, internal_replay=3, internal_planning=4,
offline_consolidation=4 (round(4*frac), clamped to [1,4]) --
non-degenerate (not all four equal), which is the runtime confirmation
that the horizon mechanism is actually biting and not silently
collapsing to a no-op. `mechanism_activation` in the manifest reports
this as a non-gating diagnostic block, checked in the smoke test before
committing to the full 30-seed grid (per this skill's Step 3.5 sample-
efficiency rule: confirm a decisive readout is non-trivially engaged
before running the full grid).

DV, THRESHOLDS, DESIGN: unchanged from V3-EXQ-869 -- see that script's
docstring for the full DV-calibration history (why
`action_object_decoder_raw_output_stats.std_by_action_dim` was chosen
over discrete first-action entropy, which saturates). Primary DV:
`mean_raw_std_by_dim`. FLOOR_DIAGNOSTIC=0.015, FLOOR_PRODUCTION=0.01,
predicted decreasing order internal_planning > external_task >
internal_replay > offline_consolidation (from mode_noise_scale's
1.3/1.0/0.5/0.3 -- the BREADTH half of the claim; horizon-depth's own
ordering, internal_planning=offline_consolidation=1.0 >
internal_replay=0.7 > external_task=0.5, is a DIFFERENT axis
(look-ahead DEPTH) and is not separately gated here -- this script tests
whether the SAME pre-registered breadth-ordering prediction survives
elite-refit now that a second mechanism is also acting on the CEM's
scoring dynamics, not whether the depth-ordering itself holds). 2
(iteration-count) x 4 (operating mode) x 30 (seed) grid, same seed
policy as 869 (shared terrain-prior weights and z_world/z_self draw
across all 8 cells per seed; independent per-cell CEM sampling seeds).

BLOCKER CHECK: none -- this script exercises only the already-built
noise-scale and horizon-depth mechanisms via
HippocampalModule.propose_trajectories(), exactly as 869 did. No new
substrate dependency.

DESIGN: 2 (iteration-count) x 4 (operating mode) x 30 (seed) grid,
structurally identical to V3-EXQ-869.

PRIMARY DV: `mean_raw_std_by_dim` (mean, across decoded action dimensions,
of the decoder's raw per-dimension output std across the n final
returned candidates). Predicted ordering (breadth axis, unchanged from
869): raw_std(internal_planning) > raw_std(external_task) >
raw_std(internal_replay) > raw_std(offline_consolidation).

PASS CRITERION: C0 (manipulation check, iters=1) -- mean pairwise raw_std
gaps between adjacent predicted-order modes each clear FLOOR_DIAGNOSTIC
(unchanged threshold from 869 -- the noise-scale-only manipulation this
checks is architecturally untouched by the horizon-depth addition, since
horizon truncation only affects CEM elite-selection SCORING, not the
initial ao_std). C1 (load-bearing, iters=3, production default) -- the
SAME three pairwise gaps each clear the SAME FLOOR_PRODUCTION as 869.
This is the retest: 869 found C1 FAIL under noise-scale-only; this run
asks whether adding the horizon-depth mechanism's influence on elite
selection changes that outcome. C2 (secondary, non-gating) -- strict
full ordering holds in a majority of individual seeds at iters=3.
`content_persistence_fraction` (mean gap at iters=3 / mean gap at
iters=1) is reported per adjacent pair, directly comparable to 869's own
value for the same statistic under noise-scale-only.

claim_ids: ['MECH-267'] (unchanged from 869; SD-MECH267-HORIZON-DEPTH is
a substrate_queue entry, not a claim -- MECH-267 remains the sole claim
under test).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_869a_mech267_mode_conditioning_content_persistence_retest.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.hippocampal.module import HippocampalModule
from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.residue.field import ResidueField
from ree_core.utils.config import E2Config, HippocampalConfig, ResidueConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EVIDENCE_ROOT = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

EXPERIMENT_PURPOSE = "evidence"

SUPERSEDES_EXQ = "V3-EXQ-869"
SUPERSEDES_EXPERIMENT_TYPE = "v3_exq_869_mech267_mode_conditioning_content_persistence"

SEEDS: List[int] = list(range(30))  # 0..29 -- identical seed set to V3-EXQ-869

# Predicted decreasing-spread BREADTH order, matching HippocampalConfig's
# default mode_noise_scale (1.3 / 1.0 / 0.5 / 0.3). Unchanged from V3-EXQ-869
# -- this is the axis the pre-registered gaps below test, not the (different)
# horizon-depth ordering.
MODES_IN_PREDICTED_ORDER: List[str] = [
    "internal_planning",
    "external_task",
    "internal_replay",
    "offline_consolidation",
]

ITER_CONDITIONS: Dict[str, int] = {
    "diagnostic_matched": 1,   # reproduces 462/465/869's single-iteration setting
    "production_default": 3,  # HippocampalConfig.num_cem_iterations default
}

WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4

# Primary DV and floors: UNCHANGED from V3-EXQ-869's own pre-registered
# calibration (pilot-run gaps 0.0546/0.0918/0.0311 at iters=1, all >3 stdev
# from zero; -0.0003/+0.0035/-0.0037 at iters=3, all indistinguishable from
# zero under noise-scale-only). Reusing the same floors here is deliberate:
# this run asks whether the SAME pre-registered bar is now cleared with the
# horizon-depth mechanism also active, not whether a new bar is.
FLOOR_DIAGNOSTIC = 0.015
FLOOR_PRODUCTION = 0.01

# Secondary per-seed ordering majority bar (non-gating corroboration).
_SEED_ORDER_MAJORITY_FRACTION = 0.5

_PRIMARY_KEY = "mean_raw_std_by_dim"
_SECONDARY_KEY = "entropy"


def _make_hippocampal(num_cem_iterations: int) -> HippocampalModule:
    e2_cfg = E2Config(
        self_dim=SELF_DIM, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)
    res_cfg = ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32, num_basis_functions=8)
    res = ResidueField(res_cfg)
    hip_cfg = HippocampalConfig(
        world_dim=WORLD_DIM, action_dim=ACTION_DIM, action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=32, horizon=HORIZON, num_candidates=NUM_CANDIDATES,
        num_cem_iterations=num_cem_iterations,
        mode_conditioning_enabled=True,
        # mode_noise_scale AND mode_horizon_scale both left at
        # HippocampalConfig defaults (1.3/1.0/0.5/0.3 and
        # 0.5/1.0/0.7/1.0 respectively) -- BOTH mechanisms are gated by
        # the same mode_conditioning_enabled=True above and both fire
        # whenever operating_mode is supplied below. This is the sole
        # substantive difference from V3-EXQ-869's identical
        # construction: at the time 869 ran, mode_horizon_scale did not
        # exist and mode_conditioning_enabled=True activated the
        # noise-scale mechanism alone.
    )
    return HippocampalModule(hip_cfg, e2, res)


def _cell_sampling_seed(seed: int, iters_label: str, mode: str) -> int:
    """Reproducible, decorrelated per-(seed, iters, mode) sampling seed.

    Identical scheme to V3-EXQ-869 (same offsets), so per-cell CEM draws
    are directly comparable seed-for-seed against the prior run.
    """
    iters_offset = {"diagnostic_matched": 0, "production_default": 500_000}[iters_label]
    mode_offset = {m: i * 7_919 for i, m in enumerate(MODES_IN_PREDICTED_ORDER)}[mode]
    return seed * 104_729 + iters_offset + mode_offset


def _run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    """One seed: shared (weights, z_world, z_self) across all 8 cells."""
    iters_labels = ["diagnostic_matched"] if dry_run else list(ITER_CONDITIONS.keys())

    cells: Dict[str, Dict[str, Any]] = {}
    for iters_label in iters_labels:
        n_iters = ITER_CONDITIONS[iters_label]

        # Same terrain-prior weights for this seed, regardless of iters_label
        # (both iteration-count conditions see the identical network).
        torch.manual_seed(seed)
        hip = _make_hippocampal(num_cem_iterations=n_iters)

        # Decorrelated but reproducible z_world/z_self draw, identical
        # across iters_labels for this seed.
        torch.manual_seed(seed + 900_000)
        z_world = torch.randn(1, WORLD_DIM)
        z_self = torch.randn(1, SELF_DIM)

        for mode in MODES_IN_PREDICTED_ORDER:
            torch.manual_seed(_cell_sampling_seed(seed, iters_label, mode))
            hip.propose_trajectories(
                z_world, z_self, num_candidates=NUM_CANDIDATES,
                operating_mode={mode: 1.0},
            )
            diag = hip.get_last_propose_diagnostics()
            raw_std = diag.get("action_object_decoder_raw_output_stats", {}).get(
                "std_by_action_dim", []
            )
            cells[f"{iters_label}::{mode}"] = {
                "iters_label": iters_label,
                "num_cem_iterations": n_iters,
                "mode": mode,
                "entropy": float(diag.get("candidate_first_action_entropy", 0.0)),
                "unique_classes": int(diag.get("candidate_unique_first_action_classes", 0)),
                "mean_raw_std_by_dim": (
                    float(sum(raw_std) / len(raw_std)) if raw_std else 0.0
                ),
                "mode_noise_scale_used": hip._last_mode_noise_scale,
                "mode_horizon_scale_used": hip._last_mode_horizon_scale,
                "effective_horizon_used": hip._last_effective_horizon,
            }

    return {"seed": seed, "cells": cells}


def _pairwise_gaps(
    cells: Dict[str, Dict[str, Any]], iters_label: str, key: str = _PRIMARY_KEY,
) -> Dict[str, float]:
    """Adjacent-mode gaps (on `key`) in predicted-decreasing order."""
    gaps = {}
    for a, b in zip(MODES_IN_PREDICTED_ORDER[:-1], MODES_IN_PREDICTED_ORDER[1:]):
        v_a = cells[f"{iters_label}::{a}"][key]
        v_b = cells[f"{iters_label}::{b}"][key]
        gaps[f"{a}_minus_{b}"] = v_a - v_b
    return gaps


def _mechanism_activation(per_seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Runtime confirmation that BOTH mode_noise_scale and
    mode_horizon_scale mechanisms actually fired -- not merely that config
    defaults imply they should. Reads the per-cell diagnostics every seed
    recorded (hip._last_mode_noise_scale / _last_mode_horizon_scale /
    _last_effective_horizon), rather than inspecting config in the
    abstract, so a future substrate change that silently no-ops one
    mechanism would show up here as a degenerate reading instead of
    passing on config inspection alone.
    """
    all_cells = [c for r in per_seed_results for c in r["cells"].values()]
    noise_scale_values = [c["mode_noise_scale_used"] for c in all_cells]
    horizon_scale_values = [c["mode_horizon_scale_used"] for c in all_cells]
    effective_horizon_values = [c["effective_horizon_used"] for c in all_cells]

    noise_scale_active = bool(all_cells) and all(v is not None for v in noise_scale_values)
    horizon_scale_active = bool(all_cells) and all(v is not None for v in horizon_scale_values)
    distinct_effective_horizons = sorted(set(
        v for v in effective_horizon_values if v is not None
    ))
    horizon_varies_by_mode = len(distinct_effective_horizons) > 1

    # Representative per-mode effective_horizon (identical across seeds for
    # a fixed mode/iters_label since it is a deterministic function of
    # config + operating_mode, not of the CEM sampling seed).
    effective_horizon_by_mode: Dict[str, Any] = {}
    if all_cells:
        for mode in MODES_IN_PREDICTED_ORDER:
            match = next(
                (c for c in all_cells if c["mode"] == mode), None
            )
            effective_horizon_by_mode[mode] = (
                match["effective_horizon_used"] if match else None
            )

    return {
        "noise_scale_active": noise_scale_active,
        "horizon_scale_active": horizon_scale_active,
        "horizon_varies_by_mode": horizon_varies_by_mode,
        "distinct_effective_horizons": distinct_effective_horizons,
        "effective_horizon_by_mode": effective_horizon_by_mode,
        "both_mechanisms_active": bool(
            noise_scale_active and horizon_scale_active and horizon_varies_by_mode
        ),
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:3] if dry_run else SEEDS
    print(
        f"[v3_exq_869a] MECH-267 mode-conditioning content-persistence RETEST "
        f"(noise-scale + horizon-depth), {len(seeds)} seed(s) "
        f"({'dry-run' if dry_run else 'full'})...",
        flush=True,
    )

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        r = _run_seed(seed, dry_run=dry_run)
        per_seed_results.append(r)
        seen_labels = sorted(set(c["iters_label"] for c in r["cells"].values()))
        for iters_label in seen_labels:
            print(f"Seed {seed} Condition {iters_label}", flush=True)
        gaps_preview = _pairwise_gaps(
            r["cells"], "diagnostic_matched" if dry_run else "production_default"
        )
        ordering_holds = all(g > 0.0 for g in gaps_preview.values())
        verdict = "PASS" if ordering_holds else "FAIL"
        print(
            f"  seed={seed} raw_std_gaps={gaps_preview} ordering_holds={ordering_holds}",
            flush=True,
        )
        print(f"verdict: {verdict}", flush=True)

    iters_labels = ["diagnostic_matched"] if dry_run else list(ITER_CONDITIONS.keys())

    mechanism_activation = _mechanism_activation(per_seed_results)
    print(
        f"[v3_exq_869a] mechanism_activation: {mechanism_activation}",
        flush=True,
    )

    # Aggregate mean gaps (primary DV) + entropy gaps (secondary DV) per
    # iters_label.
    mean_gaps: Dict[str, Dict[str, float]] = {}
    entropy_gaps: Dict[str, Dict[str, float]] = {}
    per_seed_gaps: Dict[str, List[Dict[str, float]]] = {label: [] for label in iters_labels}
    for label in iters_labels:
        all_gaps = [_pairwise_gaps(r["cells"], label, _PRIMARY_KEY) for r in per_seed_results]
        per_seed_gaps[label] = all_gaps
        keys = all_gaps[0].keys() if all_gaps else []
        mean_gaps[label] = {k: statistics.fmean(g[k] for g in all_gaps) for k in keys}

        all_entropy_gaps = [
            _pairwise_gaps(r["cells"], label, _SECONDARY_KEY) for r in per_seed_results
        ]
        entropy_gaps[label] = {
            k: statistics.fmean(g[k] for g in all_entropy_gaps) for k in keys
        }

    # Non-degeneracy: the diagnostic-matched condition must show SOME spread
    # on the primary DV (not every mode collapsed to an identical value) --
    # else the decoder / candidate pool saturated and nothing downstream is
    # informative about the claim.
    diag_gaps = mean_gaps.get("diagnostic_matched", {})
    diag_values = [
        c[_PRIMARY_KEY]
        for r in per_seed_results
        for key, c in r["cells"].items()
        if c["iters_label"] == "diagnostic_matched"
    ]
    value_spread = (max(diag_values) - min(diag_values)) if diag_values else 0.0
    non_degenerate = value_spread > 1e-6

    c0_diagnostic_manipulation_check = non_degenerate and all(
        g >= FLOOR_DIAGNOSTIC for g in diag_gaps.values()
    )

    if dry_run:
        # Smoke: only the diagnostic-matched condition ran (see iters_labels
        # above); report the manipulation check as the outcome driver.
        outcome = "PASS" if (
            c0_diagnostic_manipulation_check and mechanism_activation["both_mechanisms_active"]
        ) else "FAIL"
        evidence_direction = (
            "supports"
            if (c0_diagnostic_manipulation_check and mechanism_activation["both_mechanisms_active"])
            else "non_contributory"
        )
        c1_production_load_bearing = None
        c2_seed_majority_ordering = None
        n_seeds_full_order = None
        interpretation_label = "dry_run_manipulation_check_only"
    else:
        prod_gaps = mean_gaps.get("production_default", {})
        c1_production_load_bearing = non_degenerate and c0_diagnostic_manipulation_check and all(
            g >= FLOOR_PRODUCTION for g in prod_gaps.values()
        )

        prod_seed_gaps = per_seed_gaps.get("production_default", [])
        n_seeds_full_order = sum(
            1 for g in prod_seed_gaps if all(v > 0.0 for v in g.values())
        )
        c2_seed_majority_ordering = (
            len(prod_seed_gaps) > 0
            and n_seeds_full_order >= max(
                1, round(len(prod_seed_gaps) * _SEED_ORDER_MAJORITY_FRACTION)
            )
        )

        if not mechanism_activation["both_mechanisms_active"]:
            outcome = "FAIL"
            evidence_direction = "non_contributory"
            interpretation_label = "mechanism_activation_check_failed_not_both_mechanisms_active"
        elif not non_degenerate:
            outcome = "FAIL"
            evidence_direction = "non_contributory"
            interpretation_label = "measurement_degenerate_no_raw_std_spread"
        elif not c0_diagnostic_manipulation_check:
            outcome = "FAIL"
            evidence_direction = "non_contributory"
            interpretation_label = "manipulation_check_failed_wiring_not_reproduced_at_content_level"
        elif c1_production_load_bearing:
            outcome = "PASS"
            evidence_direction = "supports"
            interpretation_label = "mode_conditioning_content_effect_persists_under_production_cem_with_both_mechanisms"
        else:
            outcome = "FAIL"
            evidence_direction = "weakens"
            interpretation_label = "mode_conditioning_content_effect_still_washed_out_with_both_mechanisms"

    elapsed = time.perf_counter() - t0
    print(
        f"[v3_exq_869a] overall: {outcome} "
        f"(non_degenerate={non_degenerate}, C0={c0_diagnostic_manipulation_check}, "
        f"C1={c1_production_load_bearing}, both_mechanisms_active="
        f"{mechanism_activation['both_mechanisms_active']}, "
        f"mean_raw_std_gaps_diagnostic={diag_gaps}, "
        f"mean_raw_std_gaps_production={mean_gaps.get('production_default')}) "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_869a_mech267_mode_conditioning_content_persistence_retest_{ts}_v3"

    persistence_fractions: Dict[str, Any] = {}
    if not dry_run:
        prod_gaps = mean_gaps.get("production_default", {})
        for k in diag_gaps:
            denom = diag_gaps.get(k, 0.0)
            persistence_fractions[k] = (
                (prod_gaps.get(k, 0.0) / denom) if abs(denom) > 1e-9 else None
            )

    full_config = {
        "seeds": seeds,
        "modes_in_predicted_order": MODES_IN_PREDICTED_ORDER,
        "iter_conditions": ITER_CONDITIONS if not dry_run else {"diagnostic_matched": 1},
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "action_dim": ACTION_DIM,
        "action_object_dim": ACTION_OBJECT_DIM,
        "num_candidates": NUM_CANDIDATES,
        "horizon": HORIZON,
        "primary_dv": _PRIMARY_KEY,
        "secondary_dv": _SECONDARY_KEY,
        "floor_diagnostic": FLOOR_DIAGNOSTIC,
        "floor_production": FLOOR_PRODUCTION,
        "seed_order_majority_fraction": _SEED_ORDER_MAJORITY_FRACTION,
        "mode_conditioning_enabled": True,
        "supersedes": SUPERSEDES_EXQ,
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_869a_mech267_mode_conditioning_content_persistence_retest",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-267"],
        "supersedes": SUPERSEDES_EXPERIMENT_TYPE,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-267": evidence_direction},
        "evidence_direction_note": (
            "Same-question retest of V3-EXQ-869 (superseded by this run): does "
            "MECH-267's mode-conditioned CEM proposal content survive "
            "HippocampalModule's multi-iteration elite-refit dynamics to "
            "produce a measurably different FINAL candidate set, now with "
            "BOTH the noise-scale mechanism (462/465/869) AND the "
            "horizon-depth mechanism (SD-MECH267-HORIZON-DEPTH, ree-v3 "
            "e0117eea8b, built 2026-08-02) active simultaneously. V3-EXQ-869 "
            "found this washed out under noise-scale-only at production "
            "num_cem_iterations=3 despite a clean iters=1 manipulation check; "
            "the confirmed autopsy attributed this to a structural "
            "implementation gap (only one of the claim's two named "
            "mechanisms was built), not to the claim being false. "
            f"mechanism_activation confirms both mechanisms fired this run: "
            f"{mechanism_activation}. C0 (iters=1 manipulation check): "
            f"{c0_diagnostic_manipulation_check}, mean gaps {diag_gaps}. "
            f"C1 (iters=3 production default, load-bearing): "
            f"{c1_production_load_bearing}, mean gaps "
            f"{mean_gaps.get('production_default')}. Same DV "
            "(action_object_decoder_raw_output_stats.std_by_action_dim), "
            "same modes-in-predicted-order (the noise-scale/breadth axis), "
            "and same pairwise-gap floors as V3-EXQ-869 -- this is a "
            "same-question re-queue, not a redesign. Still tests only the "
            "exploration-breadth-by-mode half of MECH-267's "
            "functional_restatement plus the newly-active look-ahead-depth "
            "half's downstream influence on elite selection; the "
            "replay-content and stability-check-content halves remain out "
            "of scope (no operating_mode consumer for those exists in "
            "ree_core/hippocampal/module.py)."
        ),
        "outcome": outcome,
        "mechanism_activation": mechanism_activation,
        "interpretation": {
            "label": interpretation_label,
            "criteria": [
                {
                    "name": "C_mechanism_activation_check",
                    "load_bearing": True,
                    "passed": bool(mechanism_activation["both_mechanisms_active"]),
                    "detail": mechanism_activation,
                },
                {
                    "name": "C0_diagnostic_manipulation_check_iters1",
                    "load_bearing": True,
                    "passed": bool(c0_diagnostic_manipulation_check),
                    "mean_gaps": diag_gaps,
                    "floor": FLOOR_DIAGNOSTIC,
                },
                {
                    "name": "C1_production_content_persistence_iters3",
                    "load_bearing": True,
                    "passed": bool(c1_production_load_bearing) if c1_production_load_bearing is not None else None,
                    "mean_gaps": mean_gaps.get("production_default"),
                    "floor": FLOOR_PRODUCTION,
                },
                {
                    "name": "C2_seed_majority_full_ordering_iters3",
                    "load_bearing": False,
                    "passed": bool(c2_seed_majority_ordering) if c2_seed_majority_ordering is not None else None,
                    "n_seeds_full_order": n_seeds_full_order,
                    "n_seeds_total": len(seeds) if not dry_run else None,
                },
            ],
            "criteria_non_degenerate": {
                "C_mechanism_activation_check": bool(len(per_seed_results) > 0),
                "C0_diagnostic_manipulation_check_iters1": bool(non_degenerate),
                "C1_production_content_persistence_iters3": bool(non_degenerate and c0_diagnostic_manipulation_check),
                "C2_seed_majority_full_ordering_iters3": bool(not dry_run and non_degenerate),
            },
            "combination_rule": (
                "outcome = FAIL/non_contributory if the mechanism-activation "
                "check fails (both mode_noise_scale and mode_horizon_scale "
                "must be measured non-None and effective_horizon must vary "
                "by mode) or if the primary DV is degenerate at iters=1 or "
                "if C0 (manipulation check) fails -- none of these are "
                "informative about the claim. Else PASS/supports iff C1 (the "
                "mean pairwise raw_std gaps at production "
                "num_cem_iterations=3 each clear FLOOR_PRODUCTION, IDENTICAL "
                "floor to V3-EXQ-869); else FAIL/weakens (both mechanisms "
                "active but the content effect still does not survive "
                "elite-refit under realistic settings). C2 is reported "
                "alongside as non-gating per-seed corroboration."
            ),
        },
        "non_degenerate": bool(non_degenerate),
        "degeneracy_reason": (
            None if non_degenerate else
            f"{_PRIMARY_KEY} spread across modes at iters=1 "
            f"(diagnostic_matched) was {value_spread:.6f} (<=1e-6) -- "
            "decoder/candidate-pool saturation, not informative about mode "
            "conditioning"
        ),
        "mean_gaps_by_condition": mean_gaps,
        "entropy_gaps_by_condition_secondary": entropy_gaps,
        "content_persistence_fraction": persistence_fractions,
        "per_seed_results": per_seed_results,
        "n_seeds": len(seeds),
        "elapsed_sec": elapsed,
        "dry_run": bool(dry_run),
    }

    out_dir = EVIDENCE_ROOT / "v3_exq_869a_mech267_mode_conditioning_content_persistence_retest"
    out_file = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
    )
    if not dry_run:
        print(f"Result written to: {out_file}", flush=True)

    return {
        "outcome": outcome,
        "manifest_path": out_file,
        "run_id": run_id,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="3 seeds, diagnostic_matched (iters=1) condition only; relocates "
             "the smoke manifest, no evidence/ write.",
    )
    args = parser.parse_args()
    _result = main(dry_run=args.dry_run)
    emit_outcome(
        outcome=_result["outcome"],
        manifest_path=_result["manifest_path"],
        run_id=_result["run_id"],
        dry_run=_result["dry_run"],
    )
    sys.exit(0 if _result["outcome"] == "PASS" else 1)
