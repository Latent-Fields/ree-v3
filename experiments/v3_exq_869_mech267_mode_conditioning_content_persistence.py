"""V3-EXQ-869: MECH-267 mode-conditioned hippocampal proposals, content-
persistence confirmer (EVIDENCE).

experiment_purpose: evidence
GOV-CONFIRM-1: MECH-267 is provisional/v3_pending:false with lit_conf 0.87
and genuine_exp_count=0 (claim_evidence.v1.json) despite two landed
diagnostic PASSes (V3-EXQ-462, V3-EXQ-465, both 2026-04-21). Both are
experiment_purpose="diagnostic" (substrate wiring checks: the operating_mode
kwarg is accepted, the CEM noise-scale multiplier equals the pre-registered
convex combination of config.mode_noise_scale, and it is a no-op when
mode_conditioning_enabled=False) and are correctly excluded from claim
confidence scoring. This script is the first experiment_purpose="evidence"
test of MECH-267's own scientific content.

CLAIM TESTED (claims.yaml MECH-267 functional_restatement, verbatim
mechanism): "proposal_trajectories(world_state, operating_mode) is the
correct signature... The operating_mode vector... biases: candidate set
(external_task: task-relevant; internal_planning: counterfactual/branching;
internal_replay: previously-committed successful; offline_consolidation:
stability-check)... Without mode conditioning, replay content contaminates
external-task selection."

WHAT 462/465 ALREADY PROVED, AND WHAT THEY DID NOT (read from
ree_core/hippocampal/module.py end to end before writing this script):
`operating_mode` is consumed in exactly ONE place in the whole 3281-line
module -- `_compute_mode_noise_scale()` -- which returns a single scalar
multiplied elementwise into `ao_std` (propose_trajectories() line ~1690:
`ao_std = ao_std * mode_scale`). `ao_mean` (the terrain-prior candidate
centre) is IDENTICAL across every mode; only the CEM sampling SPREAD around
that fixed mean differs. So the mechanism, as implemented, can only ever
produce "same centre, different spread" proposal sets -- it cannot produce
the claim's "replays past-successful trajectories" content (no memory/
history lookup is wired to operating_mode anywhere) or "stability-check"
content (no distinct offline_consolidation code path exists beyond a lower
noise-scale constant). Those two content-shaping predictions stay out of
scope here exactly as the implementation_note says (deferred to EXP-0155's
blocked cue_action_proj forward path and to the unbuilt forced-replay
injection hook -- see BLOCKER CHECK below); this script tests only the
"exploration breadth by mode" half of the claim, which IS what is built.

462's own UC3/UC4/UC5 and 465's UC3/UC5 read `ao_std`'s scalar directly,
using `num_cem_iterations=1` in `_make_hippocampal()`. That is a real but
narrow proof: it shows the FIRST CEM iteration's sampling std is
mode-conditioned exactly as designed. It says nothing about whether that
effect survives to the trajectories HippocampalModule actually RETURNS
under realistic operation, because propose_trajectories() REFITS ao_std
from the elite pool's own empirical spread after every iteration
(`ao_std = self._stack_std(elite_ao_tensor) + 1e-6`, module.py ~1866) --
from iteration 2 onward, ao_std has nothing to do with the original
mode_scale multiplier. HippocampalConfig.num_cem_iterations defaults to 3
in production (ree_core/utils/config.py:1869), not 1. So the open,
falsifiable, and previously untested question is: does the mode-conditioned
noise-scale wiring 462/465 validated actually produce a measurably
different FINAL candidate set under the CEM's own multi-iteration
elite-refit dynamics -- the set E3 actually receives -- or is it washed out
before iteration 3?

DV TYPE: representational / wall-independent (per the workset why_now,
which asks for exactly this and to self-route substrate_not_ready_requeue
if only a behavioral DV were available). This script never touches action
selection, env reward, or REEAgent -- it calls
HippocampalModule.propose_trajectories() directly and reads the CONTENT
diversity of the trajectories it returns via the module's own existing
`get_last_propose_diagnostics()` accessor. Reachable today with no
cue_action_proj / forced-replay hook.

DV CALIBRATION (done during authoring, per this skill's "verify thresholds
match the actual distribution" rule -- see the module docstring's final
paragraph for the pre-registration this produced): `_summarize_trajectories`
exposes two candidate content-diversity readouts on the FINAL returned
trajectory list. `candidate_first_action_entropy` (Shannon entropy of the
discrete first-decoded-action argmax class) was tried first and is
DISCARDED as the primary DV: a 30-seed pilot run showed it SATURATES going
from external_task (mode_scale 1.0) to internal_planning (1.3) -- mean gap
-0.009 nats, i.e. statistically indistinguishable from zero even at
iters=1, because both are already near the ln(4)=1.386 entropy ceiling.
`action_object_decoder_raw_output_stats.std_by_action_dim` (mean over
action dims; the CONTINUOUS pre-argmax decoder output spread across
candidates) does NOT saturate: the same 30-seed pilot at iters=1 shows a
clean, fully monotonic ordering in the predicted direction in 30/30 seeds
(internal_planning 0.250 > external_task 0.195 > internal_replay 0.103 >
offline_consolidation 0.072; every adjacent gap >3 stdev from zero), and at
iters=3 (production default) that ordering collapses to 0/30 seeds with
every gap statistically indistinguishable from zero (means -0.0003 /
+0.0035 / -0.0037, stdevs ~0.012-0.017). This is the primary DV below.
Entropy is retained as a secondary, non-gating, reported readout (it is
still informative about the discrete-action-class picture, just not a
reliable pairwise-floor gate at the top of the predicted order).

BLOCKER CHECK (workset step 5): searched for the two behavioural successors
named in the MECH-267 implementation_note. EXP-0155 (cue_action_proj
forward-path diagnostic, for the full EXP-0158 cross-mode rule-binding env
episodes) is still blocked -- V3-EXQ-449 confirmed cue_action_proj
ungrounded (zero-gradient consumer collapse; WORKSPACE_STATE 2026-04-21),
and the fix (SD-055 use_differentiable_cem) is itself a live, separate,
un-landed substrate thread (claims.yaml SD-055-adjacent entry ~line 35474).
The forced-replay injection hook into the E3/hippocampal tick (for the full
EXP-0161 3-arm intrusion test) has never been built -- no
"forced_replay"/"forced-replay" hit anywhere in ree-v3/ or REE_assembly/
outside the 462/465/proposal notes themselves. Neither blocker is touched
by this script: it needs no cue-indexed action projection and no injection
hook, because it reads the CEM's own already-built elite-refit dynamics
directly.

DESIGN: 2 (iteration-count) x 4 (operating mode) x 30 (seed) grid. For each
seed, ONE HippocampalModule is built (torch.manual_seed(seed) immediately
before construction, so terrain-prior weights are a pure function of seed)
and ONE (z_world, z_self) pair is drawn (a decorrelated but reproducible
seed offset), shared across ALL 8 cells for that seed -- isolating mode and
iteration-count as the only manipulated variables. Iteration counts:
`iters=1` (`diagnostic_matched`) reproduces 462/465's diagnostic setting as
an in-script manipulation check; `iters=3` (`production_default`) is
HippocampalConfig's production default and is the load-bearing test. Each
mode is pinned to probability 1.0 (matching 462/465's UC3/UC5 mode pins)
with a distinct, reproducible per-cell sampling seed so CEM draws are
independent across modes within a cell.

PRIMARY DV: `mean_raw_std_by_dim` (mean, across decoded action dimensions,
of the decoder's raw per-dimension output std across the n final returned
candidates). Predicted ordering (from config.mode_noise_scale defaults:
internal_planning=1.3, external_task=1.0, internal_replay=0.5,
offline_consolidation=0.3, and the claim's own "broader (exploration-
biased)" / "tighter (consolidative)" / "tightest (low-amplitude
consolidation)" language, config.py:1996-1998):
raw_std(internal_planning) > raw_std(external_task) > raw_std(internal_replay)
> raw_std(offline_consolidation).

PASS CRITERION: C0 (manipulation check, iters=1) -- mean pairwise raw_std
gaps between adjacent predicted-order modes each clear FLOOR_DIAGNOSTIC.
C1 (load-bearing, iters=3, production default) -- the SAME three pairwise
gaps each clear the smaller FLOOR_PRODUCTION (compression from elite-refit
is expected; a genuine non-zero gap surviving it is not). C2 (secondary,
non-gating) -- strict full ordering (all three gaps > 0, no floor) holds in
a majority of individual seeds at iters=3, guarding the C1 means against a
few outlier seeds. Also reported, non-gating: `content_persistence_fraction`
per adjacent pair = mean(gap at iters=3) / mean(gap at iters=1) -- how much
of the wiring-level effect survives to the returned candidate set,
informative regardless of PASS/FAIL; and the discrete-entropy gaps
(secondary DV) for both conditions.

claim_ids: ['MECH-267'] (single claim; SD-032a is the operating_mode
SOURCE, not tested here -- matches the claim_ids-accuracy rule).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_869_mech267_mode_conditioning_content_persistence.py [--dry-run]

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

SEEDS: List[int] = list(range(30))  # 0..29

# Predicted decreasing-spread order, matching HippocampalConfig's default
# mode_noise_scale (1.3 / 1.0 / 0.5 / 0.3) and the claim's own
# broader->tighter->tightest language.
MODES_IN_PREDICTED_ORDER: List[str] = [
    "internal_planning",
    "external_task",
    "internal_replay",
    "offline_consolidation",
]

ITER_CONDITIONS: Dict[str, int] = {
    "diagnostic_matched": 1,   # reproduces 462/465's single-iteration setting
    "production_default": 3,  # HippocampalConfig.num_cem_iterations default
}

WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4

# Primary DV: continuous decoder-output spread (does NOT saturate -- see
# module docstring DV CALIBRATION section). Pairwise gap floors calibrated
# against a 30-seed pilot: diagnostic_matched (iters=1) mean gaps were
# 0.0546 / 0.0918 / 0.0311 with stdevs 0.0168 / 0.0161 / 0.0074 (every gap
# >3 stdev from zero); production_default (iters=3) mean gaps were
# -0.0003 / +0.0035 / -0.0037 with stdevs ~0.012-0.017 (every gap
# indistinguishable from zero). Floors sit with margin inside those bands.
FLOOR_DIAGNOSTIC = 0.015   # ~2 stdev below the smallest observed pilot gap
FLOOR_PRODUCTION = 0.01    # modest bar; pilot means don't approach it

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
        # mode_noise_scale left at HippocampalConfig defaults (1.3/1.0/0.5/0.3).
    )
    return HippocampalModule(hip_cfg, e2, res)


def _cell_sampling_seed(seed: int, iters_label: str, mode: str) -> int:
    """Reproducible, decorrelated per-(seed, iters, mode) sampling seed."""
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


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:3] if dry_run else SEEDS
    print(
        f"[v3_exq_869] MECH-267 mode-conditioning content-persistence, "
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
        outcome = "PASS" if c0_diagnostic_manipulation_check else "FAIL"
        evidence_direction = "supports" if c0_diagnostic_manipulation_check else "non_contributory"
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

        if not non_degenerate:
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
            interpretation_label = "mode_conditioning_content_effect_persists_under_production_cem"
        else:
            outcome = "FAIL"
            evidence_direction = "weakens"
            interpretation_label = "mode_conditioning_content_effect_washed_out_by_elite_refit"

    elapsed = time.perf_counter() - t0
    print(
        f"[v3_exq_869] overall: {outcome} "
        f"(non_degenerate={non_degenerate}, C0={c0_diagnostic_manipulation_check}, "
        f"C1={c1_production_load_bearing}, mean_raw_std_gaps_diagnostic={diag_gaps}, "
        f"mean_raw_std_gaps_production={mean_gaps.get('production_default')}) "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_869_mech267_mode_conditioning_content_persistence_{ts}_v3"

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
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_869_mech267_mode_conditioning_content_persistence",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-267"],
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {"MECH-267": evidence_direction},
        "evidence_direction_note": (
            "Representational/wall-independent confirmer testing whether "
            "MECH-267's already-wiring-validated (V3-EXQ-462/465) per-mode "
            "CEM noise-scale conditioning survives HippocampalModule's own "
            "multi-iteration elite-refit dynamics to produce a measurably "
            "different FINAL candidate set (the trajectories E3 actually "
            "receives), vs. 462/465's num_cem_iterations=1 setting where no "
            "refit dilution occurs. Primary DV is the continuous decoder-"
            "output spread (raw_output_stats.std_by_action_dim), chosen over "
            "discrete first-action entropy after a pilot run showed entropy "
            "saturates near its ceiling at the top of the predicted order "
            "(see module docstring DV CALIBRATION). C0 (iters=1 manipulation "
            f"check): {c0_diagnostic_manipulation_check}, mean gaps {diag_gaps}. "
            f"C1 (iters=3 production default, load-bearing): "
            f"{c1_production_load_bearing}, mean gaps "
            f"{mean_gaps.get('production_default')}. "
            "Tests only the exploration-breadth-by-mode half of MECH-267's "
            "functional_restatement -- the replay-content and stability-"
            "check-content halves are out of scope (no operating_mode "
            "consumer beyond the CEM noise-scale multiplier exists anywhere "
            "in ree_core/hippocampal/module.py; those remain deferred to "
            "EXP-0155/forced-replay-hook per the implementation_note)."
        ),
        "outcome": outcome,
        "interpretation": {
            "label": interpretation_label,
            "criteria": [
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
                "C0_diagnostic_manipulation_check_iters1": bool(non_degenerate),
                "C1_production_content_persistence_iters3": bool(non_degenerate and c0_diagnostic_manipulation_check),
                "C2_seed_majority_full_ordering_iters3": bool(not dry_run and non_degenerate),
            },
            "combination_rule": (
                "outcome = FAIL/non_contributory if the primary DV is "
                "degenerate (no spread across modes at iters=1) or if C0 "
                "(the manipulation check reproducing 462/465's wiring "
                "finding at the content level) fails -- neither case is "
                "informative about the claim. Else PASS/supports iff C1 "
                "(the mean pairwise raw_std gaps at production "
                "num_cem_iterations=3 each clear FLOOR_PRODUCTION); else "
                "FAIL/weakens (wiring confirmed but the content effect does "
                "not survive elite-refit under realistic settings). C2 is "
                "reported alongside as non-gating per-seed corroboration."
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

    out_dir = EVIDENCE_ROOT / "v3_exq_869_mech267_mode_conditioning_content_persistence"
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
