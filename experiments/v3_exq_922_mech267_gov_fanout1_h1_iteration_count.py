"""V3-EXQ-922: MECH-267 mode-conditioning content-persistence, GOV-FANOUT-1
H1 leg (iteration-count discrimination) (DIAGNOSTIC).

WHY THIS RUN EXISTS: V3-EXQ-869 (noise-scale mechanism only) and V3-EXQ-869a
(noise-scale + horizon-depth, both mechanisms active) each found a clean C0
manipulation check at num_cem_iterations=1 (mode-conditioned proposal
content-spread differs by mode, gaps clear FLOOR_DIAGNOSTIC) but a washed-out
C1 at the production default num_cem_iterations=3 (gaps collapse below
FLOOR_PRODUCTION, 0/30 seeds in 869 and repeated in 869a). Both runs were
adjudicated non-falsifying (confirmed failure_autopsy_V3-EXQ-869_2026-08-02,
failure_autopsy_V3-EXQ-869a_2026-08-03): 869 routed to
/implement-substrate (the horizon-depth mechanism was missing), 869a
confirmed the wash-out persists even with that mechanism built. 869a's own
fanout_recommendation (targets[0].fanout_recommendation in the confirmed
autopsy JSON) opened a 3-leg GOV-FANOUT-1 discrimination portfolio rather
than a third same-question re-letter, reasoning that a single best-guess
re-pose risks compounding the same confound a second time:

  H1 (THIS RUN): the wash-out is iteration-count DEPENDENT -- content
      persists at some intermediate iteration count and only collapses
      somewhere between 1 and 3. Untested: only iters=1 and iters=3 have
      ever been measured (869/869a both skipped iters=2 entirely).
  H2: the CEM elite-selection value function itself needs an explicit
      mode-dependent term (not merely the horizon-depth scoring WINDOW,
      which reads an already mode-independent score). NEEDS A SUBSTRATE
      BUILD -- out of scope for this /queue-experiment session.
  H3: mode-independence is structural to shared elite-refit; hard-
      partitioning candidate pools per mode (no cross-mode elite mixing
      during refit) would preserve differentiation. NEEDS A SUBSTRATE
      BUILD -- out of scope for this /queue-experiment session.

This script is H1 ONLY. H1's null (pre-registered in the autopsy's own
suggested_probes[0]): "gaps at iters=2 are as flat as iters=3." H1 needs no
substrate build -- ree_core/hippocampal/module.py's num_cem_iterations is a
plain int consumed by `for _iteration in range(self.config.num_cem_iterations)`
(module.py:1899, ree_core/utils/config.py:1975 default=3); no code anywhere
assumes the value is only ever 1 or 3. Confirmed by direct source read before
authoring this script (Step 2.5a empirical probe -- see BLOCKER CHECK below);
H2 and H3 by contrast each name a concrete missing mechanism (a new
mode-dependent value-function term / a hard-partitioned elite-refit path)
that does not exist in ree_core today, which is why only H1 is queued here.

WHY THIS IS `diagnostic`, NOT `evidence` (unlike its 869/869a predecessors):
869 and 869a each tested MECH-267's own predicted content-persistence effect
directly against a single production configuration. This run instead
discriminates BETWEEN CAUSAL EXPLANATIONS for an already-established FAIL
(is the collapse iteration-count-dependent, or is mode-independence baked in
regardless of iteration count) -- root-cause discrimination is the
`diagnostic` case per this skill's EXPERIMENT_PURPOSE guidance. Excluded
from governance confidence/conflict scoring; requires a confirmed
/failure-autopsy adjudication (not merely this script's self-routed
`interpretation.label`) before any governance action is taken on it.

WHAT CHANGED VS 869a, WHAT DID NOT: HippocampalModule/HippocampalConfig
construction, seed policy (30 seeds, identical seed integers and per-cell
sampling-seed formula), modes-in-predicted-order, primary DV
(action_object_decoder_raw_output_stats.std_by_action_dim ->
mean_raw_std_by_dim), and both pairwise-gap floors (FLOOR_DIAGNOSTIC=0.015,
FLOOR_PRODUCTION=0.01) are IDENTICAL to 869/869a -- this is the same
underlying probe, extended along ONE new axis. The only change: a THIRD
iteration-count condition, num_cem_iterations=2, inserted between the
existing iters=1 (manipulation check) and iters=3 (production, already
known FAIL) conditions. mode_horizon_scale is left at the live
HippocampalConfig defaults exactly as in 869a, so both named mechanisms
(noise-scale + horizon-depth) remain simultaneously active at every
iteration count -- this run does not re-litigate 869a's own finding, it
asks whether a DIFFERENT iteration count than production changes it.

MECHANISM ACTIVATION CHECK: identical instrumentation to 869a
(mode_noise_scale_used / mode_horizon_scale_used / effective_horizon_used
recorded per cell from hip._last_mode_noise_scale /
hip._last_mode_horizon_scale / hip._last_effective_horizon), reported as a
non-gating `mechanism_activation` diagnostic block and also promoted to a
readiness-kind `interpretation.preconditions[]` entry (see Diagnostic
adjudication gate in this skill) since this is now a diagnostic-purpose
script.

BLOCKER CHECK (Step 2.5 substrate-readiness + Step 2.5a empirical probe):
none. HippocampalConfig.num_cem_iterations is documented IMPLEMENTED
(ree-v3/CLAUDE.md, SD-034/MECH-267 sections) and, per the source read above,
takes any positive int with no special-casing of 1 or 3 -- num_cem_iterations=2
exercises the identical code path as 869/869a with one fewer/more elite-refit
pass. No new substrate dependency; both consumed mechanisms
(mode_noise_scale, mode_horizon_scale) are already built and validated
active by 869a's own mechanism_activation block.

RE-DERIVE BRAKE (Step 2.5b): MECH-267's autopsy count on this exact
same-granularity question (production-settings content-persistence,
non_contributory/competence_implementation_gap) is 2 (869, 869a) --
AT the RE_DERIVE_BRAKE_THRESHOLD default of 2. The brake is explicitly NOT
a bar to this run: 869a's own confirmed autopsy is the producer-side release
that authorizes exactly this fan-out (targets[0].routing = "queue-experiment",
targets[0].recommended_substrate_queue_entry.action = "none" -- no substrate
gate on the response, a GOV-FANOUT-1 portfolio is the prescribed route
instead of a same-question re-letter). This leg tests a DIFFERENT iteration
count than either braked run (iters=2, vs 869/869a's iters=1+3), attacking a
different design axis (iteration-count, per the autopsy's own axis
labelling) -- exactly the "not braked" carve-out ("a commitment-free read of
the same claim, or a diagnostic whose purpose is to discriminate WHY the
ceiling holds -- none of these is the re-derive loop").

DESIGN: 3 (iteration-count: 1/2/3) x 4 (operating mode) x 30 (seed) grid --
one condition wider than 869a's 2x4x30, otherwise structurally identical.

PRIMARY DV: `mean_raw_std_by_dim` (mean, across decoded action dimensions,
of the decoder's raw per-dimension output std across the n final returned
candidates). Predicted ordering (breadth axis, unchanged from 869/869a):
raw_std(internal_planning) > raw_std(external_task) >
raw_std(internal_replay) > raw_std(offline_consolidation).

DISCRIMINATION CRITERIA (H1 test, this run's load-bearing question):
  C0 (readiness/manipulation check, iters=1) -- SAME as 869/869a: mean
      pairwise raw_std gaps each clear FLOOR_DIAGNOSTIC. Must pass for
      anything downstream to be informative.
  C_H1 (load-bearing, iters=2, THE NEW TEST) -- mean pairwise raw_std gaps
      each clear FLOOR_PRODUCTION (the SAME floor 869/869a used to call
      iters=3 washed out). PASS here = H1 SUPPORTED (content persists at
      iters=2 even though it does not at iters=3 -- the wash-out IS
      iteration-count dependent). FAIL here = H1's own pre-registered null
      is met ("gaps at iters=2 are as flat as iters=3") -- H1 REFUTED,
      informative in exactly the sense GOV-FANOUT-1 asks each leg's null to
      be, not merely a wasted run.
  C1 (context, iters=3) -- SAME floor/statistic as 869a, reported for
      direct seed-for-seed comparability. Not load-bearing here (869a
      already established this reading); included so a reader can see
      whether adding one more iteration-count point changes the iters=3
      reading at all (it should not, since nothing about the iters=3 cells
      differs from 869a's construction).
  C2 (secondary, non-gating) -- fraction of seeds where the full descending
      ordering holds at iters=2, reported alongside the existing iters=3
      figure for comparison.

claim_ids: ['MECH-267'] (context tag only -- experiment_purpose=diagnostic
excludes this run from governance confidence/conflict scoring per this
skill's EXPERIMENT_PURPOSE convention; a confirmed /failure-autopsy
adjudication is required before governance acts on this run's
self-routed interpretation.label).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_922_mech267_gov_fanout1_h1_iteration_count.py [--dry-run]

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

EXPERIMENT_PURPOSE = "diagnostic"

RELATED_EXQ = ["V3-EXQ-869", "V3-EXQ-869a"]

SEEDS: List[int] = list(range(30))  # 0..29 -- identical seed set to V3-EXQ-869/869a

# Predicted decreasing-spread BREADTH order, matching HippocampalConfig's
# default mode_noise_scale (1.3 / 1.0 / 0.5 / 0.3). Unchanged from 869/869a.
MODES_IN_PREDICTED_ORDER: List[str] = [
    "internal_planning",
    "external_task",
    "internal_replay",
    "offline_consolidation",
]

ITER_CONDITIONS: Dict[str, int] = {
    "diagnostic_matched": 1,   # reproduces 462/465/869/869a's single-iteration setting
    "h1_probe": 2,             # NEW: the untested intermediate rung this leg exists to measure
    "production_default": 3,  # HippocampalConfig.num_cem_iterations default
}

WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4

# Primary DV and floors: UNCHANGED from V3-EXQ-869/869a's pre-registered
# calibration. Reusing the same floors here is deliberate -- this run asks
# whether the SAME pre-registered bar is cleared at a DIFFERENT iteration
# count, not whether a new bar is appropriate.
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
        # HippocampalConfig defaults -- identical construction to 869a, so
        # both mechanisms fire at every iteration count tested here.
    )
    return HippocampalModule(hip_cfg, e2, res)


def _cell_sampling_seed(seed: int, iters_label: str, mode: str) -> int:
    """Reproducible, decorrelated per-(seed, iters, mode) sampling seed.

    Same offset scheme as 869/869a for the two shared labels
    (diagnostic_matched, production_default); the new h1_probe label gets
    its own disjoint offset so it cannot alias either predecessor's draws.
    """
    iters_offset = {
        "diagnostic_matched": 0,
        "h1_probe": 250_000,
        "production_default": 500_000,
    }[iters_label]
    mode_offset = {m: i * 7_919 for i, m in enumerate(MODES_IN_PREDICTED_ORDER)}[mode]
    return seed * 104_729 + iters_offset + mode_offset


def _run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    """One seed: shared (weights, z_world, z_self) across all 12 cells."""
    iters_labels = ["diagnostic_matched", "h1_probe"] if dry_run else list(ITER_CONDITIONS.keys())

    cells: Dict[str, Dict[str, Any]] = {}
    for iters_label in iters_labels:
        n_iters = ITER_CONDITIONS[iters_label]

        # Same terrain-prior weights for this seed, regardless of iters_label.
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
    mode_horizon_scale mechanisms actually fired at every iteration count
    tested -- identical instrumentation to 869a, extended to cover the new
    h1_probe condition automatically (it iterates every recorded cell).
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

    effective_horizon_by_mode: Dict[str, Any] = {}
    if all_cells:
        for mode in MODES_IN_PREDICTED_ORDER:
            match = next((c for c in all_cells if c["mode"] == mode), None)
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
        f"[v3_exq_922] MECH-267 GOV-FANOUT-1 H1 (iteration-count) leg, "
        f"{len(seeds)} seed(s) ({'dry-run' if dry_run else 'full'})...",
        flush=True,
    )

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        r = _run_seed(seed, dry_run=dry_run)
        per_seed_results.append(r)
        seen_labels = sorted(set(c["iters_label"] for c in r["cells"].values()))
        for iters_label in seen_labels:
            print(f"Seed {seed} Condition {iters_label}", flush=True)
        probe_label = "h1_probe" if (dry_run or True) else "h1_probe"
        gaps_preview = _pairwise_gaps(r["cells"], probe_label)
        ordering_holds = all(g > 0.0 for g in gaps_preview.values())
        verdict = "PASS" if ordering_holds else "FAIL"
        print(
            f"  seed={seed} h1_probe_raw_std_gaps={gaps_preview} "
            f"ordering_holds={ordering_holds}",
            flush=True,
        )
        print(f"verdict: {verdict}", flush=True)

    iters_labels = (
        ["diagnostic_matched", "h1_probe"] if dry_run else list(ITER_CONDITIONS.keys())
    )

    mechanism_activation = _mechanism_activation(per_seed_results)
    print(f"[v3_exq_922] mechanism_activation: {mechanism_activation}", flush=True)

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

    # Non-degeneracy: the diagnostic-matched (iters=1) condition must show
    # SOME spread on the primary DV -- else nothing downstream is
    # informative, regardless of what iters=2/3 report.
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

    readiness_preconditions = [
        {
            "name": "both_mechanisms_active",
            "description": (
                "mode_noise_scale and mode_horizon_scale both measured "
                "non-None, and effective_horizon varies by mode -- confirms "
                "both named MECH-267 mechanisms fired at runtime, not just "
                "per config inspection."
            ),
            "measured": bool(mechanism_activation["both_mechanisms_active"]),
            "threshold": True,
            "met": bool(mechanism_activation["both_mechanisms_active"]),
            "control": "runtime diagnostics on every cell, all iteration counts",
        },
        {
            "name": "diagnostic_manipulation_check_iters1",
            "description": (
                "Same C0 gate as 869/869a: mean pairwise raw_std gaps at "
                "iters=1 each clear FLOOR_DIAGNOSTIC on a known-working "
                "manipulation (mode-conditioned proposal generation itself, "
                "before any elite-refit)."
            ),
            "measured": min(diag_gaps.values()) if diag_gaps else 0.0,
            "threshold": FLOOR_DIAGNOSTIC,
            "direction": "lower",
            "met": bool(c0_diagnostic_manipulation_check),
            "control": "iters=1, the condition 869/869a already validated clean",
        },
    ]

    if dry_run:
        h1_gaps = mean_gaps.get("h1_probe", {})
        h1_supported_smoke = non_degenerate and all(g >= FLOOR_PRODUCTION for g in h1_gaps.values())
        outcome = "PASS" if (
            c0_diagnostic_manipulation_check and mechanism_activation["both_mechanisms_active"]
        ) else "FAIL"
        evidence_direction = "non_contributory"
        interpretation_label = "dry_run_h1_probe_smoke_only"
        c_h1_load_bearing = h1_supported_smoke
        c1_context_iters3 = None
        c2_seed_majority_ordering = None
        n_seeds_full_order = None
    else:
        h1_gaps = mean_gaps.get("h1_probe", {})
        c_h1_load_bearing = non_degenerate and c0_diagnostic_manipulation_check and all(
            g >= FLOOR_PRODUCTION for g in h1_gaps.values()
        )

        prod_gaps = mean_gaps.get("production_default", {})
        c1_context_iters3 = non_degenerate and c0_diagnostic_manipulation_check and all(
            g >= FLOOR_PRODUCTION for g in prod_gaps.values()
        )

        h1_seed_gaps = per_seed_gaps.get("h1_probe", [])
        n_seeds_full_order = sum(
            1 for g in h1_seed_gaps if all(v > 0.0 for v in g.values())
        )
        c2_seed_majority_ordering = (
            len(h1_seed_gaps) > 0
            and n_seeds_full_order >= max(
                1, round(len(h1_seed_gaps) * _SEED_ORDER_MAJORITY_FRACTION)
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
        elif c_h1_load_bearing:
            outcome = "PASS"
            evidence_direction = "non_contributory"
            interpretation_label = "h1_supported_wash_out_is_iteration_count_dependent"
        else:
            outcome = "FAIL"
            evidence_direction = "non_contributory"
            interpretation_label = "h1_refuted_gaps_at_iters2_as_flat_as_iters3"

    elapsed = time.perf_counter() - t0
    print(
        f"[v3_exq_922] overall: {outcome} label={interpretation_label} "
        f"(non_degenerate={non_degenerate}, C0={c0_diagnostic_manipulation_check}, "
        f"C_H1(iters=2)={c_h1_load_bearing}, C1_context(iters=3)={c1_context_iters3}, "
        f"both_mechanisms_active={mechanism_activation['both_mechanisms_active']}, "
        f"mean_raw_std_gaps_diagnostic={diag_gaps}, "
        f"mean_raw_std_gaps_h1_probe={mean_gaps.get('h1_probe')}, "
        f"mean_raw_std_gaps_production={mean_gaps.get('production_default')}) "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_922_mech267_gov_fanout1_h1_iteration_count_{ts}_v3"

    persistence_fractions: Dict[str, Any] = {}
    if not dry_run:
        prod_gaps = mean_gaps.get("production_default", {})
        h1_gaps_final = mean_gaps.get("h1_probe", {})
        for k in diag_gaps:
            denom = diag_gaps.get(k, 0.0)
            persistence_fractions[k] = {
                "iters2_over_iters1": (
                    (h1_gaps_final.get(k, 0.0) / denom) if abs(denom) > 1e-9 else None
                ),
                "iters3_over_iters1": (
                    (prod_gaps.get(k, 0.0) / denom) if abs(denom) > 1e-9 else None
                ),
            }

    full_config = {
        "seeds": seeds,
        "modes_in_predicted_order": MODES_IN_PREDICTED_ORDER,
        "iter_conditions": (
            {"diagnostic_matched": 1, "h1_probe": 2} if dry_run else ITER_CONDITIONS
        ),
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
        "related_exq": RELATED_EXQ,
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_922_mech267_gov_fanout1_h1_iteration_count",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-267"],
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-267": evidence_direction},
        "evidence_direction_note": (
            "DIAGNOSTIC (excluded from governance confidence/conflict "
            "scoring). GOV-FANOUT-1 H1 leg of the 3-leg discrimination "
            "portfolio routed by confirmed failure_autopsy_V3-EXQ-869a_"
            "2026-08-03 (targets[0].fanout_recommendation): is the "
            "MECH-267 mode-conditioning content-persistence wash-out "
            "(869/869a: clean at iters=1, washed out at production "
            "iters=3) iteration-count DEPENDENT? Tests a THIRD "
            "iteration-count point, num_cem_iterations=2, never before "
            "measured. mechanism_activation confirms both named "
            f"mechanisms fired: {mechanism_activation}. C0 (iters=1 "
            f"manipulation check): {c0_diagnostic_manipulation_check}, "
            f"mean gaps {diag_gaps}. C_H1 (iters=2, the load-bearing "
            f"question this leg exists to answer): {c_h1_load_bearing}, "
            f"mean gaps {mean_gaps.get('h1_probe')}. C1 (iters=3, "
            f"context/comparability to 869a): {c1_context_iters3}, mean "
            f"gaps {mean_gaps.get('production_default')}. "
            f"interpretation.label={interpretation_label} is a HYPOTHESIS "
            "about the discrimination outcome, not a verdict -- requires "
            "a confirmed /failure-autopsy adjudication before governance "
            "acts on it. Same DV, same modes-in-predicted-order, same "
            "pairwise-gap floors, same seed set as V3-EXQ-869/869a; H2 "
            "(mode-aware CEM value-function term) and H3 (hard-"
            "partitioned candidate pools) remain unqueued -- both need a "
            "substrate build not performed by this run."
        ),
        "outcome": outcome,
        "mechanism_activation": mechanism_activation,
        "interpretation": {
            "label": interpretation_label,
            "preconditions": readiness_preconditions,
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
                    "name": "C_H1_iteration_count_probe_iters2",
                    "load_bearing": True,
                    "passed": bool(c_h1_load_bearing) if c_h1_load_bearing is not None else None,
                    "mean_gaps": mean_gaps.get("h1_probe"),
                    "floor": FLOOR_PRODUCTION,
                    "note": (
                        "THE H1 TEST. PASS = content persists at iters=2 "
                        "(wash-out is iteration-count dependent). FAIL = "
                        "H1's own pre-registered null met (gaps at iters=2 "
                        "as flat as iters=3)."
                    ),
                },
                {
                    "name": "C1_context_production_iters3",
                    "load_bearing": False,
                    "passed": bool(c1_context_iters3) if c1_context_iters3 is not None else None,
                    "mean_gaps": mean_gaps.get("production_default"),
                    "floor": FLOOR_PRODUCTION,
                    "note": "Context only -- already established FAIL by 869a; reported for direct comparability.",
                },
                {
                    "name": "C2_seed_majority_full_ordering_iters2",
                    "load_bearing": False,
                    "passed": bool(c2_seed_majority_ordering) if c2_seed_majority_ordering is not None else None,
                    "n_seeds_full_order": n_seeds_full_order,
                    "n_seeds_total": len(seeds) if not dry_run else None,
                },
            ],
            "criteria_non_degenerate": {
                "C_mechanism_activation_check": bool(len(per_seed_results) > 0),
                "C0_diagnostic_manipulation_check_iters1": bool(non_degenerate),
                "C_H1_iteration_count_probe_iters2": bool(non_degenerate and c0_diagnostic_manipulation_check),
                "C1_context_production_iters3": bool(non_degenerate and c0_diagnostic_manipulation_check),
                "C2_seed_majority_full_ordering_iters2": bool(not dry_run and non_degenerate),
            },
            "combination_rule": (
                "outcome = FAIL/non_contributory (this is a diagnostic; "
                "evidence_direction is always non_contributory regardless "
                "of which way the discrimination resolves -- the finding "
                "is which HYPOTHESIS is favoured, not a claim-supporting "
                "or claim-weakening reading) if the mechanism-activation "
                "check fails, or the primary DV is degenerate at iters=1, "
                "or C0 (manipulation check) fails -- none of these are "
                "informative about H1. Else: interpretation.label = "
                "h1_supported_... (outcome PASS) iff C_H1 (mean pairwise "
                "raw_std gaps at iters=2 each clear FLOOR_PRODUCTION); "
                "else h1_refuted_... (outcome FAIL). C1 and C2 are "
                "reported alongside as non-gating context."
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

    out_dir = EVIDENCE_ROOT / "v3_exq_922_mech267_gov_fanout1_h1_iteration_count"
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
        help="3 seeds, diagnostic_matched (iters=1) + h1_probe (iters=2) "
             "conditions only; relocates the smoke manifest, no evidence/ write.",
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
