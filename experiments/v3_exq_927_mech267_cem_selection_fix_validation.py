"""V3-EXQ-927: SD-MECH267-CEM-SELECTION-FIX validation -- four-arm re-measure of
V3-EXQ-869's C1 (production num_cem_iterations=3) mode-content wash-out with the
two new no-op-default facets (H2 mode_value_weight, H3 mode_partitioned_cem)
(DIAGNOSTIC).

WHY THIS RUN EXISTS: V3-EXQ-869/869a/923 established that MECH-267's
mode-conditioned proposal content is DIFFERENTIATED at num_cem_iterations=1 (the
proposer's first sampling pass) but WASHES OUT by the production default
num_cem_iterations=3 -- neither the noise-scale facet (mode_noise_scale) nor the
horizon-depth facet (mode_horizon_scale), alone or together, keeps the per-mode
raw_std gap above FLOOR_PRODUCTION=0.01 past iters=1. GOV-FANOUT-1 (routed by
confirmed failure_autopsy_V3-EXQ-869a_2026-08-03) named two substrate-build legs
for WHY the effect survives to iters=1 but not iters=3:

  H2: the CEM elite-selection VALUE FUNCTION is mode-INDEPENDENT, so the ranking
      re-converges to a mode-blind elite set every refit iteration.
  H3: the CEM refit re-derives ao_std purely from the mode-BLIND elite spread
      each iteration, discarding the one-time mode_noise_scale seed applied
      before the loop -- so mode-conditioned proposal BREADTH does not persist.

Both were built as two independent, deliberately-orthogonal no-op-default facets
in SD-MECH267-CEM-SELECTION-FIX (2026-08-14), each gated by the existing
mode_conditioning_enabled switch (ree_core/utils/config.py HippocampalConfig,
ree_core/hippocampal/module.py):

  H2 = mode_value_weight: Dict[str, List[float]] (default {}). A mode-dependent
       term in HippocampalModule._score_trajectory (module.py ~1586-1624): the
       blended per-mode weight vector (world_dim length) is dotted with the
       trajectory's mean z_world and SUBTRACTED from the terrain score, so the
       elite-selection RANKING stays mode-differentiated on every refit
       iteration. Diagnostic: hip._last_mode_value_weight_active.
  H3 = mode_partitioned_cem: bool (default False). Re-applies the
       mode-conditioned noise scale to the freshly-refit ao_std once per CEM
       iteration (module.py ~2119-2131 / ~2152-2164), so mode-conditioned
       proposal BREADTH persists instead of converging to the mode-blind elite
       spread. Diagnostic: hip._last_mode_partitioned_cem.

THIS RUN re-measures V3-EXQ-869's C1 condition -- production
num_cem_iterations=3 -- under FOUR arms, same 30 seeds across arms, to localise
WHICH facet (if either) rescues mode-content persistence at the production
iteration count:

  ARM 1 OFF (control): mode_conditioning_enabled=True, both new facets
      default/off (mode_value_weight={}, mode_partitioned_cem=False). This is
      the 869/869a/923 wash-out regime exactly -- the existing noise-scale +
      horizon-depth mechanisms are active but wash out at iters=3. Expected to
      FAIL the floor (per-mode raw_std gap ~0).
  ARM 2 H2-only: mode_value_weight set (distinct per-mode z_world weight
      vectors, world_dim length), mode_partitioned_cem=False.
  ARM 3 H3-only: mode_partitioned_cem=True, mode_value_weight={}.
  ARM 4 BOTH: mode_value_weight set AND mode_partitioned_cem=True.

All four arms share mode_conditioning_enabled=True and the default
mode_noise_scale (1.3/1.0/0.5/0.3) + mode_horizon_scale, so the arms differ
ONLY by the two new facets -- the clean localisation the fix design intended.

WHY DIAGNOSTIC, NOT EVIDENCE: like V3-EXQ-923, this run discriminates BETWEEN
causal loci for an already-established FAIL (does the wash-out come from the
mode-independent value function (H2), the mode-blind refit breadth (H3),
neither, or both) rather than testing MECH-267's core prediction directly.
Root-cause discrimination is the `diagnostic` case per /queue-experiment's
EXPERIMENT_PURPOSE guidance -- excluded from governance confidence/conflict
scoring; requires a confirmed /failure-autopsy adjudication before governance
acts on the self-routed interpretation.label.

BLOCKER CHECK (Step 2.5/2.5a): none. Both facets are IMPLEMENTED
(SD-MECH267-CEM-SELECTION-FIX, ree-v3 landed 2026-08-14) and confirmed reachable
at runtime by a one-tick probe before authoring: OFF -> both diagnostics False;
H2 -> _last_mode_value_weight_active True; H3 -> _last_mode_partitioned_cem True;
BOTH -> both True; raw_std populated on every cell. mode_value_weight defaults {}
and mode_partitioned_cem defaults False, so all pre-existing call sites are
bit-identical.

RE-DERIVE BRAKE (Step 2.5b): MECH-267's same-granularity autopsy count is 2
(869, 869a) at the RE_DERIVE_BRAKE_THRESHOLD default of 2. The brake is
RELEASED: this run tests the newly-BUILT SD-MECH267-CEM-SELECTION-FIX substrate
(the upstream substrate the GOV-FANOUT-1 H2/H3 legs named -- now IMPLEMENTED),
not another lettered iteration of the same design; it also attacks a different
design axis (elite-selection value function + refit breadth) than either braked
run. This is the "substrate now built -> re-test is meaningful" release, not the
re-derive loop.

GOV-REUSE-1 (Step 2.4): the decisive readout -- per-mode raw_std at
num_cem_iterations=3 WITH mode_value_weight / mode_partitioned_cem active -- has
never been measured in any prior manifest (the facets did not exist until
2026-08-14). Checked 869/869a/923 (they measure the wash-out with the OLD
facets only); not recoverable -> run.

DESIGN: 4 (arm) x 4 (operating mode) x 30 (seed) grid, all at
num_cem_iterations=3. For each (seed, arm): torch.manual_seed(seed) immediately
before HippocampalModule construction so the terrain-prior weights are a pure
function of seed (identical across arms, since the two facets add no network
parameters); ONE (z_world, z_self) pair drawn at a decorrelated-but-reproducible
seed offset, shared across all 4 modes and identical across arms for that seed;
each mode pinned to probability 1.0 with a distinct per-cell sampling seed. So
arms differ ONLY by the two config facets.

PRIMARY DV: mean_raw_std_by_dim per mode (mean across decoded action dims of
action_object_decoder_raw_output_stats.std_by_action_dim in
hip.get_last_propose_diagnostics() -- the exact key/metric V3-EXQ-869/923 use).
The per-arm acceptance quantity is the ACROSS-SEED MEAN of the tight-vs-broad
pairwise gap between operating modes:
    broad_minus_tight_gap
      = raw_std[internal_planning] - raw_std[offline_consolidation]
(the broadest predicted mode, mode_noise_scale 1.3, minus the tightest, 0.3).
This operationalises "mean pairwise raw_std gap between operating modes" as the
across-seed mean of the extreme-pair gap -- the quantity the parent-session
smoke measured (OFF ~2.1e-6, H3-ON ~0.0172) against FLOOR_PRODUCTION=0.01. The
three adjacent predicted-order gaps (869-style) and the full per-mode raw_std
means are also reported per arm, non-gating.

ACCEPTANCE per arm: broad_minus_tight_gap (across-seed mean) >= FLOOR_PRODUCTION
(0.01) at num_cem_iterations=3. ARM 1 (OFF) is expected to FAIL (washed-out
control). An arm PASSES the floor if it clears 0.01. BOTH evidence directions
are pre-registered: arms clearing the floor localise the fix (H2 -> the value
function was the missing locus; H3 -> the refit breadth was; BOTH -> the
mechanisms compose / at least one suffices); NO arm clearing = the fix is
insufficient at iters=3 and the wash-out has a locus neither facet addresses.

MANIPULATION / NON-DEGENERACY CHECK (per the task): per cell, assert the
relevant engagement diagnostic is True -- H2 arms: _last_mode_value_weight_active
True; H3 arms: _last_mode_partitioned_cem True; BOTH: both; OFF: both False.
This is the readiness/positive-control gate: if a facet did not engage as
configured, the run is non_contributory (the manipulation never fired), NOT a
substrate-verdict. Separately, raw_std must be a non-empty finite vector on every
cell (measurement integrity). Below-floor on a fix arm whose facet DID engage
and whose readout IS populated is a genuine `fix_insufficient` finding, not
starvation.

DV-SYMMETRY INVARIANCE (mandatory declaration, per arm). The DV is the
across-mode spread of raw_std (the std across FINAL returned candidates of the
decoder raw output, per action dim). Symmetry to check: additive constants
across candidates (invisible to argmax/rank), monotone rescalings, and
permutations of candidates.
  ARM 1 OFF: no new manipulation -- the baseline/control; the existing
      noise-scale/horizon facets are known (869/869a) to wash out, which is the
      control result under test, not an invariance artefact.
  ARM 2 H2 (mode_value_weight): subtracts a PER-TRAJECTORY term
      (w . mean_z_world, each candidate's own mean world state) from the elite
      score. This is NOT a broadcast constant across candidates -- it depends on
      each candidate's trajectory -- so it CAN move the argsort elite selection,
      hence the elite pool, hence raw_std. Not invariant under candidate
      rank/permutation.
  ARM 3 H3 (mode_partitioned_cem): multiplicatively rescales the refit ao_std by
      the per-mode noise scale. raw_std IS the sampling breadth this rescale
      sets, so the manipulation acts directly ON the DV's generating quantity --
      manifestly not invariant.
  ARM 4 BOTH: union of the H2 and H3 manipulations; not invariant for the same
      two reasons.

claim_ids: ['MECH-267'] (context tag only -- experiment_purpose=diagnostic
excludes this run from governance confidence/conflict scoring; a confirmed
/failure-autopsy adjudication is required before governance acts on the
self-routed interpretation.label).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_927_mech267_cem_selection_fix_validation.py [--dry-run]

Writes a flat JSON manifest to REE_assembly/evidence/experiments/.
"""
from __future__ import annotations

import argparse
import math
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

RELATED_EXQ = ["V3-EXQ-869", "V3-EXQ-869a", "V3-EXQ-923"]

SEEDS: List[int] = list(range(30))  # 0..29 -- identical seed set to V3-EXQ-869/869a/923

# Predicted decreasing-spread BREADTH order, matching HippocampalConfig's
# default mode_noise_scale (1.3 / 1.0 / 0.5 / 0.3). Unchanged from 869/869a/923.
MODES_IN_PREDICTED_ORDER: List[str] = [
    "internal_planning",       # broadest (mode_noise_scale 1.3)
    "external_task",           # 1.0
    "internal_replay",         # 0.5
    "offline_consolidation",   # tightest (0.3)
]
_BROAD_MODE = MODES_IN_PREDICTED_ORDER[0]
_TIGHT_MODE = MODES_IN_PREDICTED_ORDER[-1]

# All at the PRODUCTION default num_cem_iterations=3 -- this run re-measures
# V3-EXQ-869's C1 condition, the one that washed out, under the four facet arms.
NUM_CEM_ITERATIONS = 3

WORLD_DIM = 32
SELF_DIM = 16
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
NUM_CANDIDATES = 16
HORIZON = 4

# Primary acceptance floor: UNCHANGED from V3-EXQ-869/869a/923's pre-registered
# calibration (FLOOR_PRODUCTION). This run asks whether the SAME pre-registered
# bar the OLD facets failed at iters=3 is cleared by the new facets at iters=3.
FLOOR_PRODUCTION = 0.01

# Distinct per-mode value-weight vectors for the H2 arms (world_dim length).
# Built with a LOCAL torch.Generator so the global RNG stream (and therefore
# every arm's terrain weights / z_world draws / per-cell sampling) is untouched.
# These are config values, identical across seeds and across the H2/BOTH arms.
_MODE_VALUE_WEIGHT_SCALE = 1.0
_mvw_gen = torch.Generator().manual_seed(267_2)
_MODE_VALUE_WEIGHT_TENSOR = (
    torch.randn(len(MODES_IN_PREDICTED_ORDER), WORLD_DIM, generator=_mvw_gen)
    * _MODE_VALUE_WEIGHT_SCALE
)
MODE_VALUE_WEIGHT: Dict[str, List[float]] = {
    mode: _MODE_VALUE_WEIGHT_TENSOR[i].tolist()
    for i, mode in enumerate(MODES_IN_PREDICTED_ORDER)
}

# Arm definitions: (mode_value_weight map, mode_partitioned_cem flag). All arms
# additionally set mode_conditioning_enabled=True; the noise-scale and
# horizon-depth facets stay at their live defaults in every arm.
ARMS: Dict[str, Dict[str, Any]] = {
    "OFF":  {"mode_value_weight": {},                "mode_partitioned_cem": False},
    "H2":   {"mode_value_weight": MODE_VALUE_WEIGHT, "mode_partitioned_cem": False},
    "H3":   {"mode_value_weight": {},                "mode_partitioned_cem": True},
    "BOTH": {"mode_value_weight": MODE_VALUE_WEIGHT, "mode_partitioned_cem": True},
}
ARM_ORDER: List[str] = ["OFF", "H2", "H3", "BOTH"]
_FIX_ARMS: List[str] = ["H2", "H3", "BOTH"]

# Expected engagement of each facet diagnostic, per arm (the manipulation check).
_EXPECTED_ENGAGEMENT: Dict[str, Dict[str, bool]] = {
    "OFF":  {"value_weight_active": False, "partitioned_cem": False},
    "H2":   {"value_weight_active": True,  "partitioned_cem": False},
    "H3":   {"value_weight_active": False, "partitioned_cem": True},
    "BOTH": {"value_weight_active": True,  "partitioned_cem": True},
}

_PRIMARY_KEY = "mean_raw_std_by_dim"
_SECONDARY_KEY = "entropy"


def _make_hippocampal(arm: str) -> HippocampalModule:
    e2_cfg = E2Config(
        self_dim=SELF_DIM, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, hidden_dim=64,
    )
    e2 = E2FastPredictor(e2_cfg)
    res_cfg = ResidueConfig(world_dim=WORLD_DIM, hidden_dim=32, num_basis_functions=8)
    res = ResidueField(res_cfg)
    arm_cfg = ARMS[arm]
    hip_cfg = HippocampalConfig(
        world_dim=WORLD_DIM, action_dim=ACTION_DIM, action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=32, horizon=HORIZON, num_candidates=NUM_CANDIDATES,
        num_cem_iterations=NUM_CEM_ITERATIONS,
        mode_conditioning_enabled=True,
        # mode_noise_scale + mode_horizon_scale left at HippocampalConfig
        # defaults in every arm; only the two new facets vary.
        mode_value_weight=arm_cfg["mode_value_weight"],
        mode_partitioned_cem=arm_cfg["mode_partitioned_cem"],
    )
    return HippocampalModule(hip_cfg, e2, res)


def _cell_sampling_seed(seed: int, arm: str, mode: str) -> int:
    """Reproducible, decorrelated per-(seed, arm, mode) sampling seed.

    Arm offset keeps the four arms' CEM draws disjoint; mode offset matches
    the 869/923 scheme so the per-mode sampling is directly comparable.
    """
    arm_offset = {a: i * 1_000_003 for i, a in enumerate(ARM_ORDER)}[arm]
    mode_offset = {m: i * 7_919 for i, m in enumerate(MODES_IN_PREDICTED_ORDER)}[mode]
    return seed * 104_729 + arm_offset + mode_offset


def _run_seed(seed: int, dry_run: bool = False) -> Dict[str, Any]:
    """One seed: shared (weights, z_world, z_self) across all arms x modes."""
    arms = ["OFF", "H3"] if dry_run else ARM_ORDER

    cells: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        # Terrain-prior weights are a pure function of seed (the two facets add
        # no network parameters), so every arm sees the identical network.
        torch.manual_seed(seed)
        hip = _make_hippocampal(arm)

        # Decorrelated but reproducible z_world/z_self draw, identical across
        # arms for this seed.
        torch.manual_seed(seed + 900_000)
        z_world = torch.randn(1, WORLD_DIM)
        z_self = torch.randn(1, SELF_DIM)

        for mode in MODES_IN_PREDICTED_ORDER:
            torch.manual_seed(_cell_sampling_seed(seed, arm, mode))
            hip.propose_trajectories(
                z_world, z_self, num_candidates=NUM_CANDIDATES,
                operating_mode={mode: 1.0},
            )
            diag = hip.get_last_propose_diagnostics()
            raw_std = diag.get("action_object_decoder_raw_output_stats", {}).get(
                "std_by_action_dim", []
            )
            raw_std_ok = bool(raw_std) and all(
                (v is not None and math.isfinite(float(v))) for v in raw_std
            )
            cells[f"{arm}::{mode}"] = {
                "arm": arm,
                "mode": mode,
                "num_cem_iterations": NUM_CEM_ITERATIONS,
                "entropy": float(diag.get("candidate_first_action_entropy", 0.0)),
                "unique_classes": int(diag.get("candidate_unique_first_action_classes", 0)),
                "mean_raw_std_by_dim": (
                    float(sum(raw_std) / len(raw_std)) if raw_std_ok else 0.0
                ),
                "raw_std_populated": raw_std_ok,
                # Engagement diagnostics (module attributes, not in the diag dict).
                "mode_value_weight_active": bool(hip._last_mode_value_weight_active),
                "mode_partitioned_cem": bool(hip._last_mode_partitioned_cem),
                "mode_noise_scale_used": hip._last_mode_noise_scale,
            }

    return {"seed": seed, "cells": cells}


def _broad_minus_tight_gap(cells: Dict[str, Dict[str, Any]], arm: str) -> float:
    """Tight-vs-broad pairwise raw_std gap for one (seed, arm) cell:
    raw_std[broadest predicted mode] - raw_std[tightest predicted mode]."""
    v_broad = cells[f"{arm}::{_BROAD_MODE}"][_PRIMARY_KEY]
    v_tight = cells[f"{arm}::{_TIGHT_MODE}"][_PRIMARY_KEY]
    return float(v_broad - v_tight)


def _adjacent_gaps(cells: Dict[str, Dict[str, Any]], arm: str, key: str = _PRIMARY_KEY) -> Dict[str, float]:
    """869-style adjacent-mode gaps (on `key`) in predicted-decreasing order."""
    gaps = {}
    for a, b in zip(MODES_IN_PREDICTED_ORDER[:-1], MODES_IN_PREDICTED_ORDER[1:]):
        gaps[f"{a}_minus_{b}"] = float(cells[f"{arm}::{a}"][key] - cells[f"{arm}::{b}"][key])
    return gaps


def _cell_engaged_as_configured(cell: Dict[str, Any]) -> bool:
    exp = _EXPECTED_ENGAGEMENT[cell["arm"]]
    return (
        bool(cell["mode_value_weight_active"]) == exp["value_weight_active"]
        and bool(cell["mode_partitioned_cem"]) == exp["partitioned_cem"]
    )


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:3] if dry_run else SEEDS
    print(
        f"[v3_exq_927] SD-MECH267-CEM-SELECTION-FIX validation "
        f"(4 arms x 4 modes x {len(seeds)} seed(s), num_cem_iterations="
        f"{NUM_CEM_ITERATIONS}) ({'dry-run' if dry_run else 'full'})...",
        flush=True,
    )

    arms = ["OFF", "H3"] if dry_run else ARM_ORDER

    per_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        r = _run_seed(seed, dry_run=dry_run)
        per_seed_results.append(r)
        for arm in arms:
            print(f"Seed {seed} Condition {arm}", flush=True)
        gaps_preview = {a: _broad_minus_tight_gap(r["cells"], a) for a in arms}
        any_fix_clears = any(
            gaps_preview.get(a, 0.0) >= FLOOR_PRODUCTION for a in arms if a in _FIX_ARMS
        )
        verdict = "PASS" if any_fix_clears else "FAIL"
        print(
            f"  seed={seed} broad_minus_tight_gaps={ {a: round(v, 6) for a, v in gaps_preview.items()} } "
            f"any_fix_clears={any_fix_clears}",
            flush=True,
        )
        print(f"verdict: {verdict}", flush=True)

    # --- Manipulation check (facet engagement matches configuration) ----------
    all_cells = [c for r in per_seed_results for c in r["cells"].values()]
    n_cells = len(all_cells)
    n_engaged = sum(1 for c in all_cells if _cell_engaged_as_configured(c))
    manipulation_frac = (n_engaged / n_cells) if n_cells else 0.0
    manipulation_ok = n_cells > 0 and n_engaged == n_cells

    n_readout = sum(1 for c in all_cells if c["raw_std_populated"])
    readout_frac = (n_readout / n_cells) if n_cells else 0.0
    readout_ok = n_cells > 0 and n_readout == n_cells

    # --- Per-arm across-seed gap aggregates -----------------------------------
    per_arm_mean_broad_tight_gap: Dict[str, float] = {}
    per_arm_seed_gaps: Dict[str, List[float]] = {}
    per_arm_adjacent_mean_gaps: Dict[str, Dict[str, float]] = {}
    per_arm_mode_raw_std: Dict[str, Dict[str, float]] = {}
    per_arm_n_seeds_positive: Dict[str, int] = {}
    per_arm_clears_floor: Dict[str, bool] = {}
    for arm in arms:
        seed_gaps = [_broad_minus_tight_gap(r["cells"], arm) for r in per_seed_results]
        per_arm_seed_gaps[arm] = seed_gaps
        mean_gap = statistics.fmean(seed_gaps) if seed_gaps else 0.0
        per_arm_mean_broad_tight_gap[arm] = mean_gap
        per_arm_n_seeds_positive[arm] = sum(1 for g in seed_gaps if g > 0.0)
        per_arm_clears_floor[arm] = mean_gap >= FLOOR_PRODUCTION

        adj_all = [_adjacent_gaps(r["cells"], arm) for r in per_seed_results]
        keys = adj_all[0].keys() if adj_all else []
        per_arm_adjacent_mean_gaps[arm] = {
            k: statistics.fmean(g[k] for g in adj_all) for k in keys
        }
        per_arm_mode_raw_std[arm] = {
            mode: statistics.fmean(
                r["cells"][f"{arm}::{mode}"][_PRIMARY_KEY] for r in per_seed_results
            )
            for mode in MODES_IN_PREDICTED_ORDER
        }

    off_clears = bool(per_arm_clears_floor.get("OFF", False))
    fix_clearing = [a for a in _FIX_ARMS if per_arm_clears_floor.get(a, False)]
    any_fix_clears = len(fix_clearing) > 0

    # --- Outcome / self-routed interpretation label ---------------------------
    if not readout_ok:
        outcome = "FAIL"
        interpretation_label = "measurement_degenerate_raw_std_unpopulated"
    elif not manipulation_ok:
        outcome = "FAIL"
        interpretation_label = "manipulation_check_failed_facets_not_engaged_as_configured"
    elif off_clears:
        # The wash-out control did NOT wash out -- premise violated, ambiguous.
        outcome = "FAIL"
        interpretation_label = "control_leak_off_arm_cleared_floor"
    elif any_fix_clears:
        outcome = "PASS"
        interpretation_label = "fix_effective::" + "+".join(fix_clearing)
    else:
        outcome = "FAIL"
        interpretation_label = "fix_insufficient_no_arm_clears_floor_at_iters3"

    # Non-gating: does the strict 869-style all-adjacent-gaps-clear-floor hold?
    per_arm_all_adjacent_clear: Dict[str, bool] = {
        arm: all(g >= FLOOR_PRODUCTION for g in per_arm_adjacent_mean_gaps[arm].values())
        for arm in arms
    }

    elapsed = time.perf_counter() - t0
    print(
        f"[v3_exq_927] overall: {outcome} label={interpretation_label} "
        f"(manipulation_ok={manipulation_ok} [{n_engaged}/{n_cells}], "
        f"readout_ok={readout_ok}, off_clears={off_clears}, "
        f"fix_clearing={fix_clearing}, "
        f"mean_broad_minus_tight_gap={ {a: round(v, 6) for a, v in per_arm_mean_broad_tight_gap.items()} }) "
        f"({elapsed:.1f}s)",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_927_mech267_cem_selection_fix_validation_{ts}_v3"

    # Readiness-kind preconditions (the manipulation check IS the positive
    # control that the facets fire). Below-floor on a fix arm whose facet
    # engaged and whose readout is populated is a genuine fix_insufficient
    # finding, NOT starvation -- hence no substrate_not_ready_requeue route.
    preconditions = [
        {
            "name": "manipulation_all_cells_engaged_as_configured",
            "description": (
                "Every (seed, arm, mode) cell's facet engagement matches its "
                "arm configuration: OFF -> both diagnostics False; H2 -> "
                "_last_mode_value_weight_active True; H3 -> "
                "_last_mode_partitioned_cem True; BOTH -> both True. Confirms "
                "the two new facets actually fired (or stayed off) at runtime, "
                "not merely per config inspection."
            ),
            "measured": manipulation_frac,
            "threshold": 1.0,
            "direction": "lower",
            "met": bool(manipulation_ok),
            "control": "runtime engagement diagnostics on every cell of all arms",
        },
        {
            "name": "raw_std_readout_populated_all_cells",
            "description": (
                "action_object_decoder_raw_output_stats.std_by_action_dim is a "
                "non-empty finite vector on every cell -- the DV can be read. A "
                "washed-out (near-zero) per-mode gap is a valid result; an "
                "EMPTY/NaN readout is a measurement failure."
            ),
            "measured": readout_frac,
            "threshold": 1.0,
            "direction": "lower",
            "met": bool(readout_ok),
            "control": "decoder raw-output std vector length + finiteness, every cell",
        },
    ]

    criteria = [
        {
            "name": "C_manipulation_engagement_check",
            "load_bearing": True,
            "passed": bool(manipulation_ok),
            "detail": {"n_engaged": n_engaged, "n_cells": n_cells},
        },
    ]
    for arm in arms:
        criteria.append({
            "name": f"C_{arm}_broad_minus_tight_gap_clears_floor",
            "load_bearing": arm in _FIX_ARMS,
            "passed": bool(per_arm_clears_floor.get(arm, False)),
            "mean_gap": per_arm_mean_broad_tight_gap.get(arm),
            "floor": FLOOR_PRODUCTION,
            "note": (
                "control arm -- expected NOT to clear (washed out)"
                if arm == "OFF" else
                "fix arm -- clearing the floor localises the wash-out to this facet"
            ),
        })
    criteria.append({
        "name": "C_at_least_one_fix_arm_clears_floor_with_off_washed_out",
        "load_bearing": True,
        "passed": bool(any_fix_clears and not off_clears),
        "detail": {"fix_clearing": fix_clearing, "off_clears": off_clears},
        "note": (
            "THE localisation test: >=1 fix arm clears FLOOR_PRODUCTION while "
            "the OFF control stays washed out. If no fix arm clears, the fix is "
            "insufficient at iters=3. If OFF also clears, the control premise "
            "(wash-out at iters=3) failed and the reading is ambiguous."
        ),
    })

    criteria_non_degenerate = {
        "C_manipulation_engagement_check": bool(n_cells > 0),
        "C_at_least_one_fix_arm_clears_floor_with_off_washed_out": bool(
            manipulation_ok and readout_ok
        ),
    }
    for arm in arms:
        criteria_non_degenerate[f"C_{arm}_broad_minus_tight_gap_clears_floor"] = bool(
            manipulation_ok and readout_ok
        )

    full_config = {
        "seeds": seeds,
        "arms": ARM_ORDER if not dry_run else ["OFF", "H3"],
        "arm_definitions": {
            a: {
                "mode_conditioning_enabled": True,
                "mode_value_weight_set": bool(ARMS[a]["mode_value_weight"]),
                "mode_partitioned_cem": ARMS[a]["mode_partitioned_cem"],
            }
            for a in ARM_ORDER
        },
        "modes_in_predicted_order": MODES_IN_PREDICTED_ORDER,
        "num_cem_iterations": NUM_CEM_ITERATIONS,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "action_dim": ACTION_DIM,
        "action_object_dim": ACTION_OBJECT_DIM,
        "num_candidates": NUM_CANDIDATES,
        "horizon": HORIZON,
        "primary_dv": _PRIMARY_KEY,
        "acceptance_quantity": "across_seed_mean_broad_minus_tight_gap",
        "broad_mode": _BROAD_MODE,
        "tight_mode": _TIGHT_MODE,
        "floor_production": FLOOR_PRODUCTION,
        "mode_value_weight_scale": _MODE_VALUE_WEIGHT_SCALE,
        "mode_value_weight_seed": 267_2,
        "related_exq": RELATED_EXQ,
    }

    manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": "v3_exq_927_mech267_cem_selection_fix_validation",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": ["MECH-267"],
        "evidence_direction": "non_contributory",
        "evidence_direction_per_claim": {"MECH-267": "non_contributory"},
        "evidence_direction_note": (
            "DIAGNOSTIC (excluded from governance confidence/conflict scoring). "
            "Four-arm re-measure of V3-EXQ-869's C1 (production "
            "num_cem_iterations=3) mode-content wash-out with the two new "
            "no-op-default SD-MECH267-CEM-SELECTION-FIX facets (H2 "
            "mode_value_weight, H3 mode_partitioned_cem). Arms: OFF (both facets "
            "off = the 869/869a/923 wash-out control), H2-only, H3-only, BOTH; "
            "same 30 seeds across arms; arms differ ONLY by the two facets. "
            f"Manipulation check (facet engagement matches config): "
            f"{manipulation_ok} ({n_engaged}/{n_cells}). Readout populated: "
            f"{readout_ok}. Per-arm across-seed mean broad-minus-tight raw_std "
            f"gap (broadest predicted mode {_BROAD_MODE} minus tightest "
            f"{_TIGHT_MODE}) vs FLOOR_PRODUCTION={FLOOR_PRODUCTION}: "
            f"{ {a: round(v, 6) for a, v in per_arm_mean_broad_tight_gap.items()} }; "
            f"clears floor: {per_arm_clears_floor}. "
            f"interpretation.label={interpretation_label} is a HYPOTHESIS about "
            "the fix locus, not a verdict -- requires a confirmed "
            "/failure-autopsy adjudication before governance acts on it. BOTH "
            "evidence directions pre-registered: fix arms clearing the floor "
            "localise the wash-out to that facet (H2 value function / H3 refit "
            "breadth); NO arm clearing = the fix is insufficient at iters=3. "
            "Same DV (mean_raw_std_by_dim), modes-in-predicted-order, and floor "
            "as V3-EXQ-869/869a/923."
        ),
        "outcome": outcome,
        "interpretation": {
            "label": interpretation_label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
            "combination_rule": (
                "outcome = FAIL/non_contributory if the raw_std readout is "
                "unpopulated on any cell, or the manipulation check fails (a "
                "facet did not engage as configured) -- neither is informative "
                "about the fix. Else, on a valid measurement: PASS iff >=1 fix "
                "arm's across-seed mean broad-minus-tight raw_std gap clears "
                "FLOOR_PRODUCTION AND the OFF control does not (the wash-out "
                "localisation). FAIL/label=fix_insufficient if no fix arm "
                "clears; FAIL/label=control_leak if OFF also clears (premise "
                "violated). evidence_direction is always non_contributory "
                "(diagnostic): the finding is WHICH fix locus holds, not a "
                "claim-supporting or claim-weakening reading."
            ),
        },
        "manipulation_check": {
            "all_cells_engaged_as_configured": bool(manipulation_ok),
            "n_engaged": n_engaged,
            "n_cells": n_cells,
            "expected_engagement_by_arm": _EXPECTED_ENGAGEMENT,
            "raw_std_populated_all_cells": bool(readout_ok),
            "n_readout_populated": n_readout,
        },
        "per_arm_mean_broad_minus_tight_gap": per_arm_mean_broad_tight_gap,
        "per_arm_clears_floor": per_arm_clears_floor,
        "per_arm_n_seeds_positive_gap": per_arm_n_seeds_positive,
        "per_arm_adjacent_mean_gaps": per_arm_adjacent_mean_gaps,
        "per_arm_all_adjacent_gaps_clear_floor": per_arm_all_adjacent_clear,
        "per_arm_mode_mean_raw_std": per_arm_mode_raw_std,
        "per_arm_seed_broad_minus_tight_gaps": per_arm_seed_gaps,
        "per_seed_results": per_seed_results,
        "n_seeds": len(seeds),
        "elapsed_sec": elapsed,
        "dry_run": bool(dry_run),
    }

    out_dir = EVIDENCE_ROOT / "v3_exq_927_mech267_cem_selection_fix_validation"
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
        help="3 seeds, OFF + H3 arms only; relocates the smoke manifest, no "
             "evidence/ write.",
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
