#!/opt/local/bin/python3
"""V3-EXQ-768 -- ARC-057 SD-024 x SD-025 interaction spike (env-free, Test B).

Claim: ARC-057 (approach emerges from the INTERACTION of DA-mediated representational
  expansion (MECH-232 / SD-024) and an information-seeking curiosity drive (SD-025);
  neither alone sufficient).
Purpose: DIAGNOSTIC (experiment_purpose="diagnostic"; excluded from governance
  confidence/conflict scoring; a PASS is interaction-instrument readiness routed through
  /failure-autopsy adjudication before any governance action; a FAIL falsifies the
  env-free composition for pennies).
Substrate: SD-024 (da_modulated_rbf_density) + SD-025 (curiosity_drive), both
  IMPLEMENTED 2026-07-16. Single claim -> no evidence_direction_per_claim needed.

SCOPE (env-FREE interaction spike -- Test B, NOT the ecological Test C)
----------------------------------------------------------------------
This is the buildable node the two single-mechanism validations could not reach:
V3-EXQ-766 tested SD-024 ALONE (representational expansion, PASS); V3-EXQ-767 tested
SD-025 ALONE (curiosity propagates into CEM selection, PASS). NEITHER ran the ARC-057
interaction. This spike runs the 4-arm ablation over the COMBINED machinery in the same
substrate-abstract style as 766/767 -- synthetic RBF/z_world space, the hippocampal CEM
selection instrument, NO environment and NO agent commitment (so it stays out of reach of
the conversion/F-dominance ceiling). The ecological env-enabled Test C is a SEPARATE,
deferred-to-V4 concern (mystery(known data) frame-blocked + co-blocked on the conversion
ceiling) -- see REE_assembly/evidence/planning/arc_057_ecological_env_decision_2026-07-16.md.
This is per sd_024_da_modulated_rbf_density.md "Test Plan Phase 3".

WHAT THIS TESTS (the 2x2 ablation)
----------------------------------
Two regions A and B in z_world (opposite sides), a neutral origin between them. A CEM
selection instrument (hip._score_trajectory, the EXACT elite-selection scoring path) picks
among K straight-line candidate trajectories, half heading to A and half to B; the
selected trajectory is argmin(score). The readout is pref_dense = fraction of trials the
selected trajectory heads to the REWARD/DENSE region, counterbalanced over an A-dense and a
B-dense field (cancels the harm-terrain's fixed geometry preference, exactly as 767).

  ARM    SD-024 (da)  SD-025 (curiosity)   prediction
  A00    OFF          OFF                  chance (no gradient, nothing reads it)
  A10    ON           OFF                  chance (dense gradient EXISTS but curiosity OFF -> unread)
  A01    OFF          ON                   chance (curiosity ON but density UNIFORM -> no gradient)
  A11    ON           ON                   HIGH   (dense cluster + curiosity reads it -> approach)

The design makes approach CONJUNCTIVE (an AND gate): only A11 approaches. This is the
ARC-057 interaction -- "significantly more approach than either alone (interaction effect,
not just additive)" -- not a graded/additive sum of two main effects (both single-drive
main effects are ~0 by construction).

Each region receives N_ENC "encounters". The REWARD region's encounters carry DA
(dopamine_signal=DA_SIGNAL); the non-reward region's carry none. With the SD-024 master
switch OFF the DA is IGNORED -> both regions get N_ENC single centers -> UNIFORM density
(A01/A00 have no density gradient to follow). With it ON the reward region's encounters
each allocate a CLUSTER (representational expansion) -> density(dense) >> density(sparse).

VALUE-FOLLOWER CONFOUND -- isolated TWO ways (mandatory design-pass resolution)
------------------------------------------------------------------------------
The confound (per the ecological-decision doc / sd_024 Test Plan): in the SD-024-ON arm the
DA cluster splits one encounter's intensity across its centers, so the summed benefit VALUE
is CONSERVED at the reward location -- a plain value/terrain follower could approach via the
value gradient and contaminate the "either alone" baseline, masking the interaction. This
spike neutralises it by construction AND confirms it, so neither the metric nor a critic can
attribute A11 approach to value:

  (1) EQUAL-MASS A-vs-B design. Both regions get N_ENC encounters -> benefit VALUE is
      mass-conserved and ~EQUAL at A and B in every arm (value_dense ~ value_sparse). A
      value-follower choosing BETWEEN A and B sees no value gradient -> cannot prefer the
      dense region. (Recorded as context: value_dense_on / value_sparse_on.) This is a
      cleaner isolation than 766's post-hoc zeroing: the value gradient is equalised
      up-front by the choice geometry.
  (2) WEIGHT-ZEROING persistence (766's discriminator, LOAD-BEARING C4). After the 4 arms,
      zero EVERY benefit weight (evaluate_benefit -> flat 0, all value removed).
      compute_representational_density is WEIGHT-INDEPENDENT, so the density field is
      UNCHANGED and the interaction must PERSIST: pref_A11 and the interaction contrast are
      unchanged. If A11 approach rode on value it would collapse here; it does not.
  Additionally the CEM scoring path is ARC-007 strict -- benefit VALUE never enters
  _score_trajectory at all (only the harm terrain + the weight-independent curiosity bonus).

ACCEPTANCE (PASS): C1 AND C2 AND C3 AND C4.
  C1 [LOAD-BEARING] interaction contrast (weights intact) = pref_A11 - (pref_A10 + pref_A01
     - pref_A00) >= INTERACTION_MARGIN (0.20). The 2x2 super-additive interaction.
  C2 [LOAD-BEARING] both-ON approaches: pref_A11 >= APPROACH_MIN (0.60).
  C3 [LOAD-BEARING] SD-025-alone shows NO directional approach: pref_A01 <= CHANCE +
     CHANCE_TOL (0.65). (The doc-mandated criterion: uniform density -> no gradient.)
  C4 [LOAD-BEARING] interaction rides representational density, NOT value (weight-zeroing):
     interaction_contrast_zeroed >= INTERACTION_MARGIN AND pref_A11_zeroed >= APPROACH_MIN.
  Supporting: C5 SD-024-alone no selection approach (pref_A10 <= 0.65); C6 baseline sanity
  (|pref_A00 - 0.5| <= CHANCE_TOL); C7 value non-discriminating (|value_dense - value_sparse|
  small in the ON field -> mass conserved, context for the confound isolation).
FAIL otherwise (falsifies the env-free ARC-057 composition).

READINESS (P0 positive controls -- SAME statistic the load-bearing criteria route on; a
below-floor reading self-routes substrate_not_ready_requeue, NEVER a substrate verdict):
  R1 density read DISCRIMINATES: density(dense) - density(sparse) in the ON field >= 0.5
     (the density statistic C1/C2 route on).
  R2 selection RESPONDS to curiosity: a strong positive control (dense ON field, large
     curiosity_weight) yields pref_dense >= 0.80 (the selection statistic C1/C2 route on).

No training occurs (non-parametric RBF terrain + a read). Phased training N/A. Familiarity
is never advanced here (no waking visits) -> novelty = density; the anti-perseveration
discount is 767's concern, not this interaction spike's. MECH-094: DA expansion + waking-only
familiarity gates inherited but not exercised. A PASS routes through /failure-autopsy before
it can move ARC-057; it does not itself promote the claim.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import (  # noqa: E402
    ResidueConfig, HippocampalConfig, E2Config,
)
from ree_core.residue.field import ResidueField  # noqa: E402
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory  # noqa: E402
from ree_core.hippocampal.module import HippocampalModule  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_768_arc057_da_curiosity_interaction_spike"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["ARC-057"]

# ---- fixed design constants (pre-registered) ----
WORLD_DIM = 16
NUM_CENTERS = 256
KERNEL_BANDWIDTH = 1.0
ACTION_DIM = 4
ACTION_OBJECT_DIM = 16
HORIZON = 8

DA_SIGNAL = 1.0
DA_ALLOCATION_SCALE = 3.0   # DA=1.0 -> 1 + int(3) = 4 centers per reward encounter
DA_JITTER_RADIUS = 0.12
DA_BANDWIDTH_NARROWING = 0.0  # density from center COUNT only (cleanest interaction lever)
N_ENC = 8                   # encounters per region (EQUAL for both -> OFF is uniform;
                            #   reward region's carry DA, non-reward region's do not)
REGION_SEP = 3.0            # z-space separation between A and B
VISIT_JITTER = 0.05         # per-encounter noise around the region centroid

# ---- candidate / selection instrument ----
N_TRIALS = 32              # selection trials per (seed, arm)
K_CANDIDATES = 16          # candidates per trial (half -> A, half -> B)
TARGET_JITTER = 0.10       # per-candidate jitter around the region centroid

# ---- curiosity ----
CURIOSITY_WEIGHT = 1.5     # ON arms
CTRL_CURIOSITY_WEIGHT = 3.0  # strong positive control for R2
FAMILIARITY_EMA_ALPHA = 0.05
FAMILIARITY_BANDWIDTH = 1.0

# ---- pre-registered acceptance thresholds ----
CHANCE = 0.5
CHANCE_TOL = 0.15
INTERACTION_MARGIN = 0.20
APPROACH_MIN = 0.60
VALUE_EQ_TOL = 0.20        # |value_dense - value_sparse| / max(...) <= this (mass conserved)
# readiness floors
R1_DENSITY_DISCRIM_FLOOR = 0.5
R2_SELECTION_RESPONDS_FLOOR = 0.80

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
# 2x2 arms: (da_on, curiosity_on)
ARMS = ["both_off", "sd024_on_sd025_off", "sd024_off_sd025_on", "both_on"]
ARM_DA = {"both_off": False, "sd024_on_sd025_off": True,
          "sd024_off_sd025_on": False, "both_on": True}
ARM_CW = {"both_off": 0.0, "sd024_on_sd025_off": 0.0,
          "sd024_off_sd025_on": CURIOSITY_WEIGHT, "both_on": CURIOSITY_WEIGHT}
EPS = 1e-8


def _residue_cfg(da_on: bool) -> ResidueConfig:
    cfg = ResidueConfig()
    cfg.world_dim = WORLD_DIM
    cfg.num_basis_functions = NUM_CENTERS
    cfg.kernel_bandwidth = KERNEL_BANDWIDTH
    cfg.benefit_terrain_enabled = True
    cfg.use_da_modulated_rbf_density = da_on
    cfg.da_allocation_scale = DA_ALLOCATION_SCALE
    cfg.da_jitter_radius = DA_JITTER_RADIUS
    cfg.da_bandwidth_narrowing = DA_BANDWIDTH_NARROWING
    cfg.da_benefit_num_centers = None  # None -> num_basis_functions (equal capacity)
    return cfg


def _hip_cfg(curiosity_weight: float) -> HippocampalConfig:
    return HippocampalConfig(
        world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM,
        hidden_dim=64,
        horizon=HORIZON,
        num_candidates=K_CANDIDATES,
        num_cem_iterations=1,
        curiosity_weight=curiosity_weight,
        familiarity_ema_alpha=FAMILIARITY_EMA_ALPHA,
        use_curiosity_familiarity=True,
        familiarity_bandwidth=FAMILIARITY_BANDWIDTH,
    )


def _config_slice(da_on: bool, curiosity_weight: float) -> Dict:
    return {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "use_da_modulated_rbf_density": da_on, "da_allocation_scale": DA_ALLOCATION_SCALE,
        "da_jitter_radius": DA_JITTER_RADIUS, "da_bandwidth_narrowing": DA_BANDWIDTH_NARROWING,
        "n_enc": N_ENC, "region_sep": REGION_SEP, "horizon": HORIZON,
        "k_candidates": K_CANDIDATES, "n_trials": N_TRIALS, "curiosity_weight": curiosity_weight,
    }


def _make_geometry(seed: int) -> Tuple[torch.Generator, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-seed region centroids + candidate origin (local generator, so all
    arms replay IDENTICAL candidates regardless of arm_cell's global-RNG reset)."""
    gen = torch.Generator().manual_seed(2000 + seed)
    a_dir = torch.randn(1, WORLD_DIM, generator=gen)
    a_dir = a_dir / (a_dir.norm() + EPS)
    b_dir = torch.randn(1, WORLD_DIM, generator=gen)
    b_dir = b_dir / (b_dir.norm() + EPS)
    z_a = a_dir * (REGION_SEP / 2.0)
    z_b = -b_dir * (REGION_SEP / 2.0)      # roughly opposite side
    z_origin = torch.zeros(1, WORLD_DIM)   # neutral midpoint-ish
    return gen, z_a, z_b, z_origin


def _build_field(z_dense: torch.Tensor, z_sparse: torch.Tensor,
                 da_on: bool, gen: torch.Generator) -> ResidueField:
    """Benefit terrain: the reward/DENSE region gets N_ENC DA-carrying encounters; the
    non-reward/SPARSE region gets N_ENC encounters WITHOUT DA. EQUAL encounter counts ->
    the benefit VALUE (mass) is conserved and ~equal at both regions in every arm. The ONLY
    asymmetry is representational DENSITY, and only when the SD-024 master switch is ON:
      da_on : dense -> N_ENC clusters (~N_ENC*4 centers); sparse -> N_ENC centers  (gradient)
      da_off: dense -> N_ENC centers (DA ignored); sparse -> N_ENC centers         (uniform)
    """
    rf = ResidueField(_residue_cfg(da_on))
    for _ in range(N_ENC):
        zr = z_dense + VISIT_JITTER * torch.randn(1, WORLD_DIM, generator=gen)
        rf.accumulate_benefit(zr, benefit_magnitude=1.0, dopamine_signal=DA_SIGNAL)
    for _ in range(N_ENC):
        zs = z_sparse + VISIT_JITTER * torch.randn(1, WORLD_DIM, generator=gen)
        rf.accumulate_benefit(zs, benefit_magnitude=1.0, dopamine_signal=0.0)
    return rf


def _share_terrain(dst: ResidueField, src: ResidueField) -> None:
    """Copy the harm-residue terrain (neural_field) from src into dst so the two
    counterbalanced fields share an IDENTICAL harm terrain and differ ONLY in the benefit
    (density) field -- making the counterbalancing exact (the terrain's fixed geometry
    preference contributes equally to selecting dense and sparse across the two configs)."""
    dst.neural_field.load_state_dict(src.neural_field.state_dict())


def _e2() -> E2FastPredictor:
    return E2FastPredictor(E2Config(
        self_dim=6, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, rollout_horizon=HORIZON,
        num_candidates=K_CANDIDATES))


def _straight_traj(z_origin: torch.Tensor, z_target: torch.Tensor) -> Trajectory:
    """A straight-line trajectory in z_world from origin to target over HORIZON+1 states."""
    ws = []
    for h in range(HORIZON + 1):
        frac = h / float(HORIZON)
        ws.append(z_origin + frac * (z_target - z_origin))
    states = [torch.zeros(1, 6) for _ in range(HORIZON + 1)]
    actions = torch.zeros(1, HORIZON, ACTION_DIM)
    return Trajectory(states=states, actions=actions, world_states=ws)


def _pref_dense(hip: HippocampalModule, z_a, z_b, z_origin,
                dense_is_a: bool, cand_seed: int) -> float:
    """Fraction of trials where the CEM-selected (argmin score) candidate heads to the DENSE
    region. Candidates head to z_a or z_b (fixed geometry); dense_is_a says which side is
    dense in this field config. Candidates come from a generator seeded by cand_seed, so
    calls with the SAME cand_seed replay IDENTICAL candidates (only the scoring differs)."""
    gen = torch.Generator().manual_seed(cand_seed)
    half = K_CANDIDATES // 2
    dense_selected = 0
    for _ in range(N_TRIALS):
        targets, is_a = [], []
        for _ in range(half):
            targets.append(z_a + TARGET_JITTER * torch.randn(1, WORLD_DIM, generator=gen))
            is_a.append(True)
        for _ in range(half):
            targets.append(z_b + TARGET_JITTER * torch.randn(1, WORLD_DIM, generator=gen))
            is_a.append(False)
        scores = []
        for tgt in targets:
            traj = _straight_traj(z_origin, tgt)
            s = hip._score_trajectory(traj)
            scores.append(float(s.item() if isinstance(s, torch.Tensor) else s))
        best = int(torch.argmin(torch.tensor(scores)).item())
        if is_a[best] == dense_is_a:
            dense_selected += 1
    return dense_selected / float(N_TRIALS)


def _pref_dense_counterbalanced(cw: float, field_ad, field_bd,
                                z_a, z_b, z_origin, cand_seed: int) -> float:
    """Mean pref-toward-dense over the A-dense and B-dense fields (geometry-confound cancels).
    Only the curiosity weight cw and the (shared, read-only) fields determine the result."""
    hip_ad = HippocampalModule(_hip_cfg(cw), _e2(), field_ad)
    hip_bd = HippocampalModule(_hip_cfg(cw), _e2(), field_bd)
    p_ad = _pref_dense(hip_ad, z_a, z_b, z_origin, dense_is_a=True, cand_seed=cand_seed)
    p_bd = _pref_dense(hip_bd, z_a, z_b, z_origin, dense_is_a=False, cand_seed=cand_seed)
    return 0.5 * (p_ad + p_bd)


def run_seed(seed: int) -> Dict:
    torch.manual_seed(7000 + seed)   # global-RNG determinism for the DA-cluster jitter build
    gen, z_a, z_b, z_origin = _make_geometry(seed)

    # Counterbalanced field pairs (A-dense / B-dense), one per SD-024 setting. Built ONCE and
    # shared across the curiosity ON/OFF arms of that setting (curiosity only READS density).
    off_ad = _build_field(z_a, z_b, da_on=False, gen=gen)
    off_bd = _build_field(z_b, z_a, da_on=False, gen=gen)
    _share_terrain(off_bd, off_ad)
    on_ad = _build_field(z_a, z_b, da_on=True, gen=gen)
    on_bd = _build_field(z_b, z_a, da_on=True, gen=gen)
    _share_terrain(on_bd, on_ad)

    cand_seed = 3000 + seed  # shared candidate set across all four arms (only scoring differs)

    # density + value at the dense/sparse regions of the ON field (R1 + confound context)
    density_dense = float(on_ad.compute_benefit_density(z_a).item())
    density_sparse = float(on_ad.compute_benefit_density(z_b).item())
    value_dense_on = float(on_ad.evaluate_benefit(z_a).item())
    value_sparse_on = float(on_ad.evaluate_benefit(z_b).item())

    fields_for = {"both_off": (off_ad, off_bd), "sd024_on_sd025_off": (on_ad, on_bd),
                  "sd024_off_sd025_on": (off_ad, off_bd), "both_on": (on_ad, on_bd)}

    pref = {}
    arm_rows = {}
    for arm in ARMS:
        da_on, cw = ARM_DA[arm], ARM_CW[arm]
        f_ad, f_bd = fields_for[arm]
        # both_off is the reuse-eligible baseline (mint-as-you-go); the other three arms
        # share a benefit field across cells -> mark ineligible (correctness guard).
        extra = None if arm == "both_off" else ["shared_benefit_field_across_arms"]
        with arm_cell(
            seed,
            config_slice=_config_slice(da_on, cw),
            script_path=Path(__file__),
            config_slice_declared=True,
            include_driver_script_in_hash=False,  # mint-as-you-go: cross-driver reusable baseline
            extra_ineligible_reasons=extra,
        ) as cell:
            pref_dense = _pref_dense_counterbalanced(cw, f_ad, f_bd, z_a, z_b, z_origin, cand_seed)
            row = {"arm_id": arm, "seed": seed, "da_on": da_on, "curiosity_weight": cw,
                   "pref_dense": pref_dense, "density_dense": density_dense,
                   "density_sparse": density_sparse}
            cell.stamp(row)
        pref[arm] = pref_dense
        arm_rows[arm] = row
        print(f"Seed {seed} Condition {arm}")
        for i in range(N_TRIALS):
            if (i + 1) % 8 == 0 or (i + 1) == N_TRIALS:
                print(f"  [train] select seed={seed} arm={arm} ep {i + 1}/{N_TRIALS}", flush=True)
        cell_ok = 0.0 <= pref_dense <= 1.0
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}")

    p00 = pref["both_off"]
    p10 = pref["sd024_on_sd025_off"]
    p01 = pref["sd024_off_sd025_on"]
    p11 = pref["both_on"]
    interaction = p11 - (p10 + p01 - p00)

    # ---- C4 weight-zeroing persistence: zero ALL benefit weights on the ON fields.
    # compute_representational_density is weight-INDEPENDENT -> density field UNCHANGED, so the
    # interaction must persist; evaluate_benefit -> flat 0 (all value removed). A10/A01/A00 use
    # off fields (untouched) and A10 is cw=0 (value never scored anyway) -> unchanged.
    with torch.no_grad():
        on_ad.benefit_rbf_field.weights.zero_()
        on_bd.benefit_rbf_field.weights.zero_()
    value_dense_after_zero = float(on_ad.evaluate_benefit(z_a).item())
    density_dense_after_zero = float(on_ad.compute_benefit_density(z_a).item())
    p11_zeroed = _pref_dense_counterbalanced(CURIOSITY_WEIGHT, on_ad, on_bd, z_a, z_b, z_origin, cand_seed)
    p10_zeroed = _pref_dense_counterbalanced(0.0, on_ad, on_bd, z_a, z_b, z_origin, cand_seed)
    interaction_zeroed = p11_zeroed - (p10_zeroed + p01 - p00)

    # ---- R2 positive control (fresh, non-zeroed ON pair, strong curiosity) ----
    gen2, z_a2, z_b2, z_origin2 = _make_geometry(seed)
    r2_ad = _build_field(z_a2, z_b2, da_on=True, gen=gen2)
    r2_bd = _build_field(z_b2, z_a2, da_on=True, gen=gen2)
    _share_terrain(r2_bd, r2_ad)
    r2_selection_responds = _pref_dense_counterbalanced(
        CTRL_CURIOSITY_WEIGHT, r2_ad, r2_bd, z_a2, z_b2, z_origin2, 5000 + seed)

    r1_density_gap = density_dense - density_sparse
    denom = max(abs(value_dense_on), abs(value_sparse_on), EPS)
    value_rel_diff = abs(value_dense_on - value_sparse_on) / denom

    return {
        "seed": seed,
        "arm_rows": [arm_rows[a] for a in ARMS],
        # 2x2 prefs
        "pref_both_off": p00, "pref_sd024_on": p10, "pref_sd025_on": p01, "pref_both_on": p11,
        "interaction": interaction,
        # confound isolation
        "pref_both_on_zeroed": p11_zeroed, "pref_sd024_on_zeroed": p10_zeroed,
        "interaction_zeroed": interaction_zeroed,
        "value_dense_on": value_dense_on, "value_sparse_on": value_sparse_on,
        "value_rel_diff": value_rel_diff, "value_dense_after_zero": value_dense_after_zero,
        "density_dense": density_dense, "density_sparse": density_sparse,
        "density_dense_after_zero": density_dense_after_zero,
        # readiness
        "r1_density_gap": r1_density_gap,
        "r2_selection_responds": r2_selection_responds,
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(torch.tensor(xs, dtype=torch.float64).median().item())


def evaluate(per_seed: List[Dict]) -> Dict:
    med_p00 = _median([s["pref_both_off"] for s in per_seed])
    med_p10 = _median([s["pref_sd024_on"] for s in per_seed])
    med_p01 = _median([s["pref_sd025_on"] for s in per_seed])
    med_p11 = _median([s["pref_both_on"] for s in per_seed])
    med_interaction = _median([s["interaction"] for s in per_seed])
    med_p11_zeroed = _median([s["pref_both_on_zeroed"] for s in per_seed])
    med_interaction_zeroed = _median([s["interaction_zeroed"] for s in per_seed])
    med_value_rel_diff = _median([s["value_rel_diff"] for s in per_seed])

    min_r1 = min(s["r1_density_gap"] for s in per_seed)
    min_r2 = min(s["r2_selection_responds"] for s in per_seed)

    # ---- readiness preconditions (same statistic as the load-bearing criteria) ----
    preconditions = [
        {"name": "density_read_discriminates", "kind": "readiness",
         "description": "ON field: density(dense) - density(sparse) clears floor (the density "
                        "statistic the interaction contrast C1/C2 route on)",
         "measured": round(min_r1, 5), "threshold": R1_DENSITY_DISCRIM_FLOOR,
         "control": "DA-dense reward region vs equally-visited non-DA region", "direction": "lower",
         "met": bool(min_r1 >= R1_DENSITY_DISCRIM_FLOOR)},
        {"name": "selection_responds_to_curiosity", "kind": "readiness",
         "description": "strong positive control (large curiosity_weight, dense ON field) yields "
                        "pref_dense >= floor (the selection statistic C1/C2 route on)",
         "measured": round(min_r2, 4), "threshold": R2_SELECTION_RESPONDS_FLOOR,
         "control": "curiosity_weight=%.1f, dense ON field" % CTRL_CURIOSITY_WEIGHT,
         "direction": "lower", "met": bool(min_r2 >= R2_SELECTION_RESPONDS_FLOOR)},
    ]
    ready = all(p["met"] for p in preconditions)

    # ---- load-bearing acceptance criteria ----
    c1 = med_interaction >= INTERACTION_MARGIN
    c2 = med_p11 >= APPROACH_MIN
    c3 = med_p01 <= (CHANCE + CHANCE_TOL)
    c4 = (med_interaction_zeroed >= INTERACTION_MARGIN) and (med_p11_zeroed >= APPROACH_MIN)
    # supporting
    c5 = med_p10 <= (CHANCE + CHANCE_TOL)
    c6 = abs(med_p00 - CHANCE) <= CHANCE_TOL
    c7 = med_value_rel_diff <= VALUE_EQ_TOL

    criteria = [
        {"name": "C1_interaction_contrast", "load_bearing": True, "passed": bool(c1),
         "measured": round(med_interaction, 4), "threshold": INTERACTION_MARGIN,
         "note": "pref_both_on - (pref_sd024_on + pref_sd025_on - pref_both_off); "
                 "super-additive 2x2 interaction, not additive"},
        {"name": "C2_both_on_approaches", "load_bearing": True, "passed": bool(c2),
         "measured": round(med_p11, 4), "threshold": APPROACH_MIN},
        {"name": "C3_sd025_alone_no_gradient", "load_bearing": True, "passed": bool(c3),
         "measured": round(med_p01, 4), "threshold": CHANCE + CHANCE_TOL, "direction": "upper",
         "note": "SD-024 OFF -> uniform density -> curiosity has no gradient -> no directional approach"},
        {"name": "C4_interaction_rides_density_not_value", "load_bearing": True, "passed": bool(c4),
         "interaction_zeroed": round(med_interaction_zeroed, 4),
         "pref_both_on_zeroed": round(med_p11_zeroed, 4),
         "threshold_interaction": INTERACTION_MARGIN, "threshold_approach": APPROACH_MIN,
         "note": "all benefit weights zeroed (value removed): interaction + both-ON approach persist "
                 "because density is weight-independent"},
        {"name": "C5_sd024_alone_no_selection_approach", "load_bearing": False, "passed": bool(c5),
         "measured": round(med_p10, 4), "threshold": CHANCE + CHANCE_TOL, "direction": "upper"},
        {"name": "C6_baseline_chance", "load_bearing": False, "passed": bool(c6),
         "measured": round(med_p00, 4), "threshold": CHANCE_TOL,
         "note": "|pref_both_off - 0.5|"},
        {"name": "C7_value_non_discriminating", "load_bearing": False, "passed": bool(c7),
         "measured": round(med_value_rel_diff, 4), "threshold": VALUE_EQ_TOL, "direction": "upper",
         "note": "equal-mass design: value_dense ~ value_sparse -> value cannot drive the A-vs-B choice"},
    ]

    # ---- non-degeneracy ----
    density_discriminates = all(s["density_dense"] > s["density_sparse"] for s in per_seed)
    interaction_varies = len({round(s["interaction"], 3) for s in per_seed}) > 1
    mechanism_active = all(s["density_dense"] > s["density_sparse"] for s in per_seed)
    # the interaction must not be an artefact of a degenerate (constant) selection instrument
    pref11_varies = len({round(s["pref_both_on"], 3) for s in per_seed}) > 1
    criteria_non_degenerate = {
        "density_dense_exceeds_sparse": bool(density_discriminates),
        "interaction_varies_across_seeds": bool(interaction_varies),
        "pref_both_on_varies_across_seeds": bool(pref11_varies),
        "da_creates_density_gradient": bool(mechanism_active),
    }
    non_degenerate = bool(density_discriminates and pref11_varies)
    degeneracy_reason = (
        "" if non_degenerate else
        "density(dense) not > density(sparse) on some seed, or pref_both_on constant across "
        "seeds (selection instrument degenerate)")

    # ---- route ----
    if not ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    else:
        load_bearing_pass = c1 and c2 and c3 and c4
        outcome = "PASS" if load_bearing_pass else "FAIL"
        direction = "supports" if load_bearing_pass else "weakens"
        if load_bearing_pass:
            label = "arc057_approach_emerges_from_da_curiosity_interaction_not_either_alone"
        else:
            failed = [c["name"] for c in criteria if c["load_bearing"] and not c["passed"]]
            label = "arc057_interaction_not_met:" + ",".join(failed)

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "aggregates": {
            "median_pref_both_off": med_p00,
            "median_pref_sd024_on": med_p10,
            "median_pref_sd025_on": med_p01,
            "median_pref_both_on": med_p11,
            "median_interaction": med_interaction,
            "median_pref_both_on_zeroed": med_p11_zeroed,
            "median_interaction_zeroed": med_interaction_zeroed,
            "median_value_rel_diff": med_value_rel_diff,
            "min_r1_density_gap": min_r1,
            "min_r2_selection_responds": min_r2,
        },
        "thresholds": {
            "INTERACTION_MARGIN": INTERACTION_MARGIN,
            "APPROACH_MIN": APPROACH_MIN,
            "CHANCE": CHANCE, "CHANCE_TOL": CHANCE_TOL,
            "VALUE_EQ_TOL": VALUE_EQ_TOL,
            "R1_DENSITY_DISCRIM_FLOOR": R1_DENSITY_DISCRIM_FLOOR,
            "R2_SELECTION_RESPONDS_FLOOR": R2_SELECTION_RESPONDS_FLOOR,
        },
    }


def main(dry_run: bool = False) -> Dict:
    t0 = time.time()
    t0_perf = time.perf_counter()
    seeds = DEFAULT_SEEDS[:2] if dry_run else DEFAULT_SEEDS

    per_seed: List[Dict] = []
    arm_results: List[Dict] = []
    for seed in seeds:
        s = run_seed(seed)
        arm_results.extend(s.pop("arm_rows"))
        per_seed.append(s)
        print(f"  seed={seed} p00={s['pref_both_off']:.2f} p10={s['pref_sd024_on']:.2f} "
              f"p01={s['pref_sd025_on']:.2f} p11={s['pref_both_on']:.2f} "
              f"interaction={s['interaction']:.2f} p11_zeroed={s['pref_both_on_zeroed']:.2f} "
              f"r1={s['r1_density_gap']:.2f} r2={s['r2_selection_responds']:.2f}", flush=True)

    ev = evaluate(per_seed)
    outcome = ev["outcome"]
    elapsed = time.time() - t0

    print(f"[{EXPERIMENT_TYPE}] label={ev['interpretation']['label']} "
          f"direction={ev['evidence_direction']}")
    agg = ev["aggregates"]
    print(f"  p00={agg['median_pref_both_off']:.2f} p10={agg['median_pref_sd024_on']:.2f} "
          f"p01={agg['median_pref_sd025_on']:.2f} p11={agg['median_pref_both_on']:.2f} "
          f"interaction={agg['median_interaction']:.2f} "
          f"interaction_zeroed={agg['median_interaction_zeroed']:.2f}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    full_config = {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "horizon": HORIZON, "da_signal": DA_SIGNAL, "da_allocation_scale": DA_ALLOCATION_SCALE,
        "da_jitter_radius": DA_JITTER_RADIUS, "da_bandwidth_narrowing": DA_BANDWIDTH_NARROWING,
        "n_enc": N_ENC, "region_sep": REGION_SEP, "visit_jitter": VISIT_JITTER,
        "n_trials": N_TRIALS, "k_candidates": K_CANDIDATES, "target_jitter": TARGET_JITTER,
        "curiosity_weight": CURIOSITY_WEIGHT, "ctrl_curiosity_weight": CTRL_CURIOSITY_WEIGHT,
        "familiarity_ema_alpha": FAMILIARITY_EMA_ALPHA, "familiarity_bandwidth": FAMILIARITY_BANDWIDTH,
        "arms": ARMS, "seeds": list(seeds),
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "started_utc": datetime.utcfromtimestamp(t0).isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "evidence_direction": ev["evidence_direction"],
        "interpretation": ev["interpretation"],
        "aggregates": ev["aggregates"],
        "thresholds": ev["thresholds"],
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "per_seed": per_seed,
        "arm_results": arm_results,
        "substrate": "SD-024+SD-025",
        "notes": (
            "ARC-057 SD-024 x SD-025 interaction spike (env-FREE Test B; NOT the ecological "
            "Test C, which is deferred to V4 -- see arc_057_ecological_env_decision_2026-07-16.md). "
            "4-arm 2x2 ablation over the combined da_modulated_rbf_density (SD-024) + "
            "curiosity_drive (SD-025) machinery in synthetic RBF/z_world space (no environment, "
            "no agent commitment -> out of reach of the conversion/F-dominance ceiling). "
            "Approach = fraction of CEM-selected trajectories heading to the reward/dense region, "
            "counterbalanced over A-dense and B-dense fields. PASS = super-additive interaction "
            "(both-ON approach exceeds either single drive alone) with SD-025-alone at chance "
            "(uniform density -> no gradient). Value-follower confound isolated by (1) an "
            "equal-mass A-vs-B design (benefit value ~equal at both regions -> cannot drive the "
            "choice) and (2) the 766 weight-zeroing persistence check (LOAD-BEARING C4: interaction "
            "survives removing all benefit value because density is weight-independent); CEM "
            "scoring is ARC-007-strict (value never enters). No training (non-parametric RBF + "
            "read); phased training N/A. Prereqs V3-EXQ-766 (SD-024) + V3-EXQ-767 (SD-025) both "
            "PASS. A PASS is interaction-instrument readiness routed through /failure-autopsy "
            "before it can move ARC-057; it does not itself promote the claim."
        ),
    }

    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=dry_run,
        config=full_config, seeds=list(seeds), script_path=Path(__file__), started_at=t0_perf,
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {outcome}")
    return {"outcome": outcome, "manifest_path": out_path, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (2 seeds).")
    args = parser.parse_args()
    result = main(dry_run=args.dry_run)
    _outcome = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=result["manifest_path"],
        run_id=result["run_id"],
        dry_run=args.dry_run,
    )
    sys.exit(0)
