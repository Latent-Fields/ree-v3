#!/opt/local/bin/python3
"""V3-EXQ-767a -- SD-025 curiosity-drive selection-bias, CONTINUOUS score-margin re-op.

Claim: SD-025 (hippocampal_module.curiosity_drive)
Purpose: DIAGNOSTIC (experiment_purpose="diagnostic"; excluded from governance
  confidence/conflict scoring; a PASS is drive-mechanism readiness, routed through
  /failure-autopsy adjudication before any governance action; a FAIL refutes the drive).
Substrate: SD-025 (curiosity_drive) on SD-024 (da_modulated_rbf_density), both
  IMPLEMENTED 2026-07-16.
Single claim -> no evidence_direction_per_claim needed.
Supersedes: v3_exq_767 (measurement re-operationalization, same scientific question).

WHY THIS ITERATION (re-operationalize a vacuous_pass -- measurement_degeneracy)
------------------------------------------------------------------------------
The predecessor V3-EXQ-767 PASSed but was adjudicated VACUOUS
(failure_autopsy_767-768-cluster_2026-07-16, confirmed; applied in governance cycle
2026-07-16d, REE_assembly commit e40362ebb7). Its load-bearing gate was the BINARY
argmin selection fraction `pref_dense`: for any nonzero curiosity weight the curiosity
term (density gap ~58-80, w=1.0) dwarfs the near-zero harm terrain and any candidate
jitter, so argmin picks a dense candidate on EVERY trial, EVERY seed -> pref_dense = 1.0
exactly, ZERO cross-seed variance. The gate carries no effect-size and cannot clear on
evidence (script's own guard flagged pref_on_varies_across_seeds=False -> non_degenerate
false). The MECHANISM is real (OFF / single-drive controls sit at 0.5 chance); only the
readout is degenerate -- a `mystery (known data)` (the data is there; the frame is wrong).

THE FIX (identical to 768a): replace the saturating binary argmin fraction with the
CONTINUOUS CEM score-margin `margin = mean_trials[min(sparse-heading score)
- min(dense-heading score)]`, which was ALREADY COMPUTED per candidate inside `_pref_dense`
and DISCARDED. CEM minimises score, so a lower dense-region score => margin > 0; the SIGN
of `margin` is EXACTLY the old binary decision, but its MAGNITUDE reflects drive strength
and is variance-bearing (it scales with w * (density gap), which already varies 58-80
across seeds). Every load-bearing criterion now rides this margin; the binary prefs are
retained as non-load-bearing context. Counterbalancing over an A-dense and a B-dense field
cancels the harm-terrain geometry: OFF counterbalances to a clean ~0 margin null, so a
small positive floor cleanly separates the drive signal, and the per-seed margins carry
the effect size. A criteria_non_degenerate guard requires margin_on to VARY across seeds
so this cannot re-flag vacuous_pass.

SCOPE (drive mechanism ONLY -- NOT the full ARC-057 ecological claim)
---------------------------------------------------------------------
The full ARC-057 approach-emergence claim (SD-024 x SD-025 interaction produces approach
with no explicit valence gradient) is ENV-CONSTRAINED (see arc_057_ecological_env_decision).
This experiment tests the DRIVE MECHANISM: does the SD-025 curiosity term PROPAGATE into
hippocampal CEM elite selection, biasing it (by a graded score-margin) toward regions of
higher representational density (the propagation MECH-111's broadcast-novelty->E3 path
could NOT achieve, EXQ-141b/590a)? And does the familiarity discount attenuate that margin
on revisit (anti-perseveration)?

WHAT THIS TESTS
---------------
A DA-dense benefit cluster (region A, high compute_local_density) and a sparse region
(region B, one center, low density) are built into a shared benefit terrain. K candidate
trajectories are scored with HippocampalModule._score_trajectory (the EXACT scoring path
CEM elite selection uses). Half the candidates head to A, half to B. curiosity ON vs OFF,
identical candidates & field. The hippocampal terrain_score is the harm-residue field
(ARC-007 strict; benefit VALUE never enters scoring); the curiosity term adds
-curiosity_weight * mean(density * (1-familiarity)), reading compute_representational_density
(WEIGHT-INDEPENDENT), so a dense region lowers the score (CEM minimises).

LEG 1 -- PROPAGATION (curiosity produces a positive dense-preferring score-margin):
  L1a margin_on = counterbalanced mean CEM score-margin (sparse - dense), curiosity ON
      >= L1A_MARGIN_ON_FLOOR (the drive yields a positive dense-preferring margin).   [LOAD-BEARING]
  L1b margin_propagation_delta = margin_on - margin_off >= L1B_MARGIN_PROP_FLOOR
      (the selection margin actually SHIFTS with the drive; OFF ~ 0).                  [LOAD-BEARING]
  L1c weight-independence: zero every benefit weight (benefit VALUE -> flat 0). density
      is weight-independent, and hippocampal scoring never reads benefit value, so the ON
      margin is UNCHANGED (relative): |margin_on_zeroed - margin_on| / |margin_on|
      <= L1C_WEIGHT_INDEP_REL_TOL.                                                     [LOAD-BEARING]
      -> the margin rides representational DENSITY, provably not a value gradient.

LEG 2 -- ANTI-PERSEVERATION (familiarity discount is functional):
  L2a repeated WAKING visits raise familiarity -> novelty decays -> the margin toward the
      familiarized region drops: antipersev_margin = counterbalanced mean drop
      >= L2A_ANTIPERSEV_MARGIN_FLOOR.                                                  [LOAD-BEARING]
  L2b MECH-094 control: familiarizing with is_waking=False (replay/sim) does NOT shift the
      margin: |margin_replay_after - margin_replay_fresh| <= L2B_REPLAY_MARGIN_TOL.    [supporting]

ACCEPTANCE (PASS): L1a AND L1b AND L1c AND L2a. FAIL otherwise (refutes the drive mechanism).

READINESS (P0 positive controls -- SAME statistic the load-bearing criteria route on -- the
continuous margin; a below-floor reading self-routes substrate_not_ready_requeue, NEVER a
substrate verdict):
  R1 density read DISCRIMINATES: density(A) - density(B) >= R1_DENSITY_DISCRIM_FLOOR (the
     density statistic the curiosity term routes on).
  R2 selection RESPONDS to curiosity: a strong positive control (dense A, sparse B, large
     curiosity_weight) yields a positive score-MARGIN >= R2_MARGIN_FLOOR (the SAME margin
     statistic L1a/L1b route on -- NOT the binary fraction).

No training occurs (non-parametric RBF terrain + a read + an EMA). Phased training N/A.
MECH-094: familiarity updates on WAKING visits only (exercised directly by L2b).
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

EXPERIMENT_TYPE = "v3_exq_767a_sd025_curiosity_drive_selection_bias_margin"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["SD-025"]
SUPERSEDES = "v3_exq_767_sd025_curiosity_drive_selection_bias"

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
N_REWARD_A = 24             # dense region A: DA-carrying reward encounters
N_SPARSE_B = 1             # sparse region B: single non-DA center
REGION_SEP = 3.0           # z-space separation between A and B
VISIT_JITTER = 0.05

# ---- candidate / selection instrument ----
N_TRIALS = 32              # selection trials per (seed, arm)
K_CANDIDATES = 16          # candidates per trial (half -> A, half -> B)
TARGET_JITTER = 0.10       # per-candidate jitter around the region centroid

# ---- curiosity + familiarity ----
CURIOSITY_WEIGHT = 1.0
FAMILIARITY_EMA_ALPHA = 0.05
FAMILIARITY_BANDWIDTH = 1.0
N_FAM_ROUNDS = 12          # times the (HORIZON+1)-point approach corridor is revisited (waking)
CTRL_CURIOSITY_WEIGHT = 3.0  # strong positive control for R2

# ---- pre-registered acceptance thresholds (CONTINUOUS CEM score-margin, score units) ----
# OFF counterbalances to a clean ~0 margin null (harm terrain shared + identical candidates),
# so a small positive absolute floor cleanly separates the drive signal; the per-seed margins
# (expected ~tens, varying 58-80 across seeds by the density gap) carry the effect size, and
# the criteria_non_degenerate margin-variance guard is the anti-vacuity mechanism.
L1A_MARGIN_ON_FLOOR = 1.0        # med margin_on: drive yields a positive dense-preferring margin
L1B_MARGIN_PROP_FLOOR = 1.0      # med(margin_on - margin_off): margin shifts with the drive
L1C_WEIGHT_INDEP_REL_TOL = 0.10  # relative change of margin under benefit-weight zeroing
L2A_ANTIPERSEV_MARGIN_FLOOR = 1.0  # med counterbalanced margin drop after WAKING familiarization
L2B_REPLAY_MARGIN_TOL = 0.50     # replay familiarization must not shift the margin (~0 expected)
# readiness floors
R1_DENSITY_DISCRIM_FLOOR = 0.5
R2_MARGIN_FLOOR = 1.0            # strong positive control yields a positive margin (SAME statistic)

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
ARMS = ["curiosity_off", "curiosity_on"]
EPS = 1e-8


def _residue_cfg() -> ResidueConfig:
    cfg = ResidueConfig()
    cfg.world_dim = WORLD_DIM
    cfg.num_basis_functions = NUM_CENTERS
    cfg.kernel_bandwidth = KERNEL_BANDWIDTH
    cfg.benefit_terrain_enabled = True
    cfg.use_da_modulated_rbf_density = True
    cfg.da_allocation_scale = DA_ALLOCATION_SCALE
    cfg.da_jitter_radius = DA_JITTER_RADIUS
    cfg.da_bandwidth_narrowing = 0.0
    cfg.da_benefit_num_centers = None
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


def _config_slice(curiosity_weight: float) -> Dict:
    return {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "use_da_modulated_rbf_density": True, "da_allocation_scale": DA_ALLOCATION_SCALE,
        "da_jitter_radius": DA_JITTER_RADIUS, "n_reward_a": N_REWARD_A, "n_sparse_b": N_SPARSE_B,
        "region_sep": REGION_SEP, "horizon": HORIZON, "k_candidates": K_CANDIDATES,
        "n_trials": N_TRIALS, "curiosity_weight": curiosity_weight,
        "familiarity_ema_alpha": FAMILIARITY_EMA_ALPHA,
    }


def _make_geometry(seed: int) -> Tuple[torch.Generator, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic per-seed region centroids + candidate origin (local generator, so
    both arms replay IDENTICAL candidates regardless of arm_cell's global-RNG reset)."""
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
                 gen: torch.Generator) -> ResidueField:
    """Build a benefit terrain: a DA-dense cluster at z_dense, a single sparse center at
    z_sparse. Built ONCE per (seed, config) and shared by both arms (curiosity only READS
    density). COUNTERBALANCING: run_seed builds BOTH an A-dense and a B-dense field, so the
    fixed geometry-preference of the (random-init) harm-residue terrain cancels when the
    margin toward the DENSE region is averaged over the two configs -- isolating the
    curiosity contribution from where the terrain happens to point."""
    rf = ResidueField(_residue_cfg())
    for _ in range(N_REWARD_A):
        zr = z_dense + VISIT_JITTER * torch.randn(1, WORLD_DIM, generator=gen)
        rf.accumulate_benefit(zr, benefit_magnitude=1.0, dopamine_signal=DA_SIGNAL)
    for _ in range(N_SPARSE_B):
        rf.accumulate_benefit(z_sparse, benefit_magnitude=1.0, dopamine_signal=0.0)
    return rf


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
                dense_is_a: bool, cand_seed: int) -> Dict[str, float]:
    """Selection readout over N_TRIALS counterbalanced trials. Returns BOTH:
      - `frac`   : fraction of trials whose CEM-selected (argmin score) candidate heads to
                   the DENSE region (the legacy binary readout -- retained as context).
      - `margin` : the CONTINUOUS CEM score-margin = mean over trials of
                   [min(sparse-heading score) - min(dense-heading score)]. CEM minimises
                   score, so this is > 0 exactly when the dense region wins the argmin, and
                   its MAGNITUDE reflects how strongly the drive prefers dense. This is the
                   variance-bearing statistic the binary `frac` discarded (767a re-op).

    Candidates head to z_a or z_b (fixed geometry); `dense_is_a` says which side is dense in
    this field config. Candidates come from a generator seeded by cand_seed, so calls with
    the SAME cand_seed replay IDENTICAL candidates (only the scoring differs across arms)."""
    gen = torch.Generator().manual_seed(cand_seed)
    half = K_CANDIDATES // 2
    dense_selected = 0
    margins: List[float] = []
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
        dense_scores = [sc for sc, a in zip(scores, is_a) if a == dense_is_a]
        sparse_scores = [sc for sc, a in zip(scores, is_a) if a != dense_is_a]
        # CEM elite = argmin; the signed margin by which the best dense candidate beats the
        # best sparse candidate (min sparse - min dense). > 0 <=> dense is the CEM elite.
        margins.append(min(sparse_scores) - min(dense_scores))
    return {"frac": dense_selected / float(N_TRIALS),
            "margin": float(sum(margins) / len(margins))}


def _build_field_two_dense(z_a: torch.Tensor, z_b: torch.Tensor,
                           gen: torch.Generator) -> ResidueField:
    """Benefit terrain with A AND B equally DA-dense. Used by LEG 2: with both regions
    equally dense+novel the curiosity competition is symmetric, so familiarizing one region
    has maximal leverage to shift the margin toward the still-novel one (anti-perseveration)."""
    rf = ResidueField(_residue_cfg())
    for z in (z_a, z_b):
        for _ in range(N_REWARD_A):
            zr = z + VISIT_JITTER * torch.randn(1, WORLD_DIM, generator=gen)
            rf.accumulate_benefit(zr, benefit_magnitude=1.0, dopamine_signal=DA_SIGNAL)
    return rf


def _corridor(z_origin: torch.Tensor, z_target: torch.Tensor) -> List[torch.Tensor]:
    """The HORIZON+1 waypoints an approach trajectory to z_target traverses (origin->target)."""
    return [z_origin + (h / float(HORIZON)) * (z_target - z_origin) for h in range(HORIZON + 1)]


def _share_terrain(dst: ResidueField, src: ResidueField) -> None:
    """Copy the harm-residue terrain (the neural_field) from src into dst so the two
    counterbalanced fields share an IDENTICAL harm terrain and differ ONLY in the benefit
    (density) field. This makes the counterbalancing exact: the terrain's fixed geometry
    preference contributes equally to both configs -> the OFF baseline margin averages to ~0
    and the propagation delta isolates the curiosity contribution."""
    dst.neural_field.load_state_dict(src.neural_field.state_dict())


def _e2() -> E2FastPredictor:
    return E2FastPredictor(E2Config(
        self_dim=6, world_dim=WORLD_DIM, action_dim=ACTION_DIM,
        action_object_dim=ACTION_OBJECT_DIM, rollout_horizon=HORIZON,
        num_candidates=K_CANDIDATES))


def _pref_dense_counterbalanced(cw: float, field_ad, field_bd,
                                z_a, z_b, z_origin, cand_seed: int) -> Dict[str, float]:
    """Mean {frac, margin}-toward-dense over the A-dense and B-dense fields (geometry-confound
    cancels; OFF margin -> ~0, the clean null)."""
    hip_ad = HippocampalModule(_hip_cfg(cw), _e2(), field_ad)
    hip_bd = HippocampalModule(_hip_cfg(cw), _e2(), field_bd)
    r_ad = _pref_dense(hip_ad, z_a, z_b, z_origin, dense_is_a=True, cand_seed=cand_seed)
    r_bd = _pref_dense(hip_bd, z_a, z_b, z_origin, dense_is_a=False, cand_seed=cand_seed)
    return {"frac": 0.5 * (r_ad["frac"] + r_bd["frac"]),
            "margin": 0.5 * (r_ad["margin"] + r_bd["margin"])}


def run_seed(seed: int) -> Dict:
    torch.manual_seed(7000 + seed)   # global-RNG determinism for the DA-cluster jitter build
    gen, z_a, z_b, z_origin = _make_geometry(seed)
    # Counterbalanced fields: one with A dense, one with B dense (cancels harm-terrain geometry).
    field_ad = _build_field(z_a, z_b, gen)   # A dense, B sparse
    field_bd = _build_field(z_b, z_a, gen)   # B dense, A sparse
    _share_terrain(field_bd, field_ad)       # identical harm terrain -> exact counterbalance
    cand_seed_l1 = 3000 + seed               # leg-1 candidate set (off / on / on-zeroed share it)

    density_dense = float(field_ad.compute_benefit_density(z_a).item())
    density_sparse = float(field_ad.compute_benefit_density(z_b).item())

    # ---- per-arm counterbalanced margin/frac toward dense (fields shared; only curiosity differs) ----
    res = {}
    arm_rows = {}
    for arm in ARMS:
        cw = CURIOSITY_WEIGHT if arm == "curiosity_on" else 0.0
        with arm_cell(
            seed,
            config_slice=_config_slice(cw),
            script_path=Path(__file__),
            config_slice_declared=True,
            include_driver_script_in_hash=False,   # mint-as-you-go: reuse-eligible OFF baseline
            extra_ineligible_reasons=(["shared_benefit_field_across_arms"]
                                      if arm == "curiosity_on" else None),
        ) as cell:
            r = _pref_dense_counterbalanced(
                cw, field_ad, field_bd, z_a, z_b, z_origin, cand_seed_l1)
            row = {"arm_id": arm, "seed": seed, "curiosity_weight": cw,
                   "pref_dense": r["frac"], "margin_dense": r["margin"],
                   "density_dense": density_dense, "density_sparse": density_sparse}
            cell.stamp(row)
        res[arm] = r
        arm_rows[arm] = row
        print(f"Seed {seed} Condition {arm}")
        for i in range(N_TRIALS):
            if (i + 1) % 8 == 0 or (i + 1) == N_TRIALS:
                print(f"  [train] select seed={seed} arm={arm} ep {i + 1}/{N_TRIALS}", flush=True)
        cell_ok = (density_dense > density_sparse) and (0.0 <= r["frac"] <= 1.0)
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}")

    margin_off, margin_on = res["curiosity_off"]["margin"], res["curiosity_on"]["margin"]
    pref_off, pref_on = res["curiosity_off"]["frac"], res["curiosity_on"]["frac"]

    # ---- L1c weight-independence: zero benefit VALUE, density unchanged -> margin unchanged.
    # hippocampal scoring never reads benefit value (ARC-007 strict); curiosity reads the
    # weight-INDEPENDENT density -> the ON margin must be unchanged by zeroing weights.
    with torch.no_grad():
        field_ad.benefit_rbf_field.weights.zero_()
        field_bd.benefit_rbf_field.weights.zero_()
    value_dense_after_zero = float(field_ad.evaluate_benefit(z_a).item())
    density_dense_after_zero = float(field_ad.compute_benefit_density(z_a).item())
    res_on_zeroed = _pref_dense_counterbalanced(
        CURIOSITY_WEIGHT, field_ad, field_bd, z_a, z_b, z_origin, cand_seed_l1)
    margin_on_zeroed = res_on_zeroed["margin"]
    pref_on_zeroed = res_on_zeroed["frac"]
    margin_weight_indep_rel = abs(margin_on_zeroed - margin_on) / max(abs(margin_on), EPS)

    # ---- LEG 2: familiarity discount (anti-perseveration) + MECH-094 replay control ----
    # TWO equally-dense regions (symmetric curiosity competition). Familiarize one region's
    # approach corridor (WAKING) and measure how far the margin toward THAT region drops --
    # counterbalanced over which region is familiarized, so the harm-terrain's fixed
    # preference cancels (the fresh margin is terrain-dominated; the DROP isolates familiarity).
    gen2, z_a2, z_b2, z_origin2 = _make_geometry(seed)
    field2 = _build_field_two_dense(z_a2, z_b2, gen2)   # A and B both dense
    cand_seed_l2 = 4000 + seed
    corridor_a = _corridor(z_origin2, z_a2)
    corridor_b = _corridor(z_origin2, z_b2)

    def _famil_shift(fam_corridor, dense_is_a: bool, waking: bool) -> Tuple[float, float]:
        """margin-toward-the-familiarized-region, fresh vs after visiting its corridor."""
        hip = HippocampalModule(_hip_cfg(CURIOSITY_WEIGHT), _e2(), field2)
        m_fresh = _pref_dense(hip, z_a2, z_b2, z_origin2,
                              dense_is_a=dense_is_a, cand_seed=cand_seed_l2)["margin"]
        for _ in range(N_FAM_ROUNDS):
            for zc in fam_corridor:
                hip.update_familiarity(zc, is_waking=waking)
        m_after = _pref_dense(hip, z_a2, z_b2, z_origin2,
                              dense_is_a=dense_is_a, cand_seed=cand_seed_l2)["margin"]
        return m_fresh, m_after

    mA_fresh, mA_after = _famil_shift(corridor_a, dense_is_a=True, waking=True)   # familiarize A
    mB_fresh, mB_after = _famil_shift(corridor_b, dense_is_a=False, waking=True)  # familiarize B
    antipersev_margin = 0.5 * ((mA_fresh - mA_after) + (mB_fresh - mB_after))

    # MECH-094 control: replay familiarization (is_waking=False) must NOT shift the margin.
    mAr_fresh, mAr_after = _famil_shift(corridor_a, dense_is_a=True, waking=False)
    replay_margin_shift = abs(mAr_fresh - mAr_after)

    # ---- READINESS positive controls (margin -- SAME statistic the load-bearing crits route on) ----
    r1_density_gap = density_dense - density_sparse
    # matched A-dense/B-sparse counterbalanced pair for the R2 selection-response control.
    field_r2a = _build_field(z_a2, z_b2, gen2)
    field_r2b = _build_field(z_b2, z_a2, gen2)
    _share_terrain(field_r2b, field_r2a)
    res_r2 = _pref_dense_counterbalanced(
        CTRL_CURIOSITY_WEIGHT, field_r2a, field_r2b, z_a2, z_b2, z_origin2, 5000 + seed)
    r2_margin = res_r2["margin"]
    r2_frac = res_r2["frac"]

    return {
        "seed": seed,
        "arm_rows": [arm_rows["curiosity_off"], arm_rows["curiosity_on"]],
        # key names kept for the evaluate()/index contract; semantics = toward DENSE region
        "density_a": density_dense, "density_b": density_sparse,
        # leg 1 CONTINUOUS margins (LOAD-BEARING)
        "margin_off": margin_off, "margin_on": margin_on,
        "margin_propagation_delta": margin_on - margin_off,
        "margin_on_zeroed": margin_on_zeroed,
        "margin_weight_indep_rel": margin_weight_indep_rel,
        # leg 1 binary prefs (CONTEXT ONLY -- not load-bearing; retained for continuity)
        "pref_a_off": pref_off, "pref_a_on": pref_on, "pref_a_on_zeroed": pref_on_zeroed,
        "propagation_delta": pref_on - pref_off,
        "value_a_after_zero": value_dense_after_zero,
        "density_dense_after_zero": density_dense_after_zero,
        # leg 2 CONTINUOUS margins (LOAD-BEARING anti-perseveration)
        "margin_a_fresh": mA_fresh, "margin_a_familiarized": mA_after,
        "margin_b_fresh": mB_fresh, "margin_b_familiarized": mB_after,
        "antipersev_margin": antipersev_margin,
        "margin_a_replay_fresh": mAr_fresh, "margin_a_replay": mAr_after,
        "replay_margin_shift": replay_margin_shift,
        # readiness
        "r1_density_gap": r1_density_gap,
        "r2_margin": r2_margin, "r2_frac": r2_frac,
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(torch.tensor(xs, dtype=torch.float64).median().item())


def evaluate(per_seed: List[Dict]) -> Dict:
    # ---- CONTINUOUS margin aggregates (load-bearing) ----
    med_margin_on = _median([s["margin_on"] for s in per_seed])
    med_margin_off = _median([s["margin_off"] for s in per_seed])
    med_margin_prop = _median([s["margin_propagation_delta"] for s in per_seed])
    med_margin_wi_rel = _median([s["margin_weight_indep_rel"] for s in per_seed])
    med_antipersev = _median([s["antipersev_margin"] for s in per_seed])
    med_replay_margin = _median([s["replay_margin_shift"] for s in per_seed])
    # ---- binary pref aggregates (context only) ----
    med_pref_on = _median([s["pref_a_on"] for s in per_seed])
    med_pref_off = _median([s["pref_a_off"] for s in per_seed])

    min_r1 = min(s["r1_density_gap"] for s in per_seed)
    min_r2 = min(s["r2_margin"] for s in per_seed)

    # ---- readiness preconditions (SAME statistic as the load-bearing criteria: margin) ----
    preconditions = [
        {"name": "density_read_discriminates", "kind": "readiness",
         "description": "density(A) - density(B) clears floor (the density statistic the "
                        "curiosity term routes on)",
         "measured": round(min_r1, 5), "threshold": R1_DENSITY_DISCRIM_FLOOR,
         "control": "DA-dense A vs single-center B", "direction": "lower",
         "met": bool(min_r1 >= R1_DENSITY_DISCRIM_FLOOR)},
        {"name": "selection_margin_responds_to_curiosity", "kind": "readiness",
         "description": "strong positive control (large curiosity_weight, dense A) yields a "
                        "positive CEM score-MARGIN >= floor (the SAME margin statistic L1a/L1b "
                        "route on)",
         "measured": round(min_r2, 4), "threshold": R2_MARGIN_FLOOR,
         "control": "curiosity_weight=%.1f, dense A vs sparse B" % CTRL_CURIOSITY_WEIGHT,
         "direction": "lower", "met": bool(min_r2 >= R2_MARGIN_FLOOR)},
    ]
    ready = all(p["met"] for p in preconditions)

    # ---- load-bearing acceptance criteria (all ride the CONTINUOUS margin) ----
    l1a = med_margin_on >= L1A_MARGIN_ON_FLOOR
    l1b = med_margin_prop >= L1B_MARGIN_PROP_FLOOR
    l1c = med_margin_wi_rel <= L1C_WEIGHT_INDEP_REL_TOL
    l2a = med_antipersev >= L2A_ANTIPERSEV_MARGIN_FLOOR
    l2b = med_replay_margin <= L2B_REPLAY_MARGIN_TOL

    criteria = [
        {"name": "L1a_margin_on", "load_bearing": True, "passed": bool(l1a),
         "measured": round(med_margin_on, 4), "threshold": L1A_MARGIN_ON_FLOOR,
         "note": "continuous CEM score-margin (sparse - dense) under curiosity; > 0 <=> dense "
                 "is the CEM elite; magnitude reflects drive strength (re-op of the saturating "
                 "binary pref)"},
        {"name": "L1b_margin_propagation_delta", "load_bearing": True, "passed": bool(l1b),
         "measured": round(med_margin_prop, 4), "threshold": L1B_MARGIN_PROP_FLOOR,
         "note": "margin_on - margin_off; OFF counterbalances to ~0 -> the margin shifts with the drive"},
        {"name": "L1c_weight_independence", "load_bearing": True, "passed": bool(l1c),
         "measured": round(med_margin_wi_rel, 4), "threshold": L1C_WEIGHT_INDEP_REL_TOL,
         "direction": "upper",
         "note": "benefit weights zeroed -> value flat; density-driven margin unchanged (relative)"},
        {"name": "L2a_antiperseveration_margin", "load_bearing": True, "passed": bool(l2a),
         "measured": round(med_antipersev, 4), "threshold": L2A_ANTIPERSEV_MARGIN_FLOOR,
         "note": "counterbalanced drop of the margin toward the familiarized region (WAKING)"},
        {"name": "L2b_mech094_replay_no_shift", "load_bearing": False, "passed": bool(l2b),
         "measured": round(med_replay_margin, 4), "threshold": L2B_REPLAY_MARGIN_TOL,
         "direction": "upper",
         "note": "is_waking=False must not raise familiarity -> margin unchanged"},
    ]

    # ---- non-degeneracy (guard on the CONTINUOUS margin -- must vary across seeds) ----
    margin_on_varies = len({round(s["margin_on"], 3) for s in per_seed}) > 1
    density_discriminates = all(s["density_a"] > s["density_b"] for s in per_seed)
    on_margin_exceeds_off = all(s["margin_on"] > s["margin_off"] for s in per_seed)
    criteria_non_degenerate = {
        "margin_on_varies_across_seeds": bool(margin_on_varies),
        "density_a_exceeds_b": bool(density_discriminates),
        "on_margin_exceeds_off": bool(on_margin_exceeds_off),
    }
    non_degenerate = bool(density_discriminates and margin_on_varies)
    degeneracy_reason = (
        "" if non_degenerate else
        "density(A) not > density(B) on some seed, or margin_on constant across seeds "
        "(continuous selection margin degenerate)")

    # ---- route ----
    if not ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    else:
        load_bearing_pass = l1a and l1b and l1c and l2a
        outcome = "PASS" if load_bearing_pass else "FAIL"
        direction = "supports" if load_bearing_pass else "weakens"
        if load_bearing_pass:
            label = "curiosity_drive_propagates_into_cem_selection_margin_with_familiarity_discount"
        else:
            failed = [c["name"] for c in criteria if c["load_bearing"] and not c["passed"]]
            label = "sd025_drive_not_met:" + ",".join(failed)

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
            "median_margin_on": med_margin_on,
            "median_margin_off": med_margin_off,
            "median_margin_propagation_delta": med_margin_prop,
            "median_margin_weight_indep_rel": med_margin_wi_rel,
            "median_antipersev_margin": med_antipersev,
            "median_replay_margin_shift": med_replay_margin,
            # binary prefs (context)
            "median_pref_a_on": med_pref_on,
            "median_pref_a_off": med_pref_off,
            "min_r1_density_gap": min_r1,
            "min_r2_margin": min_r2,
        },
        "thresholds": {
            "L1A_MARGIN_ON_FLOOR": L1A_MARGIN_ON_FLOOR,
            "L1B_MARGIN_PROP_FLOOR": L1B_MARGIN_PROP_FLOOR,
            "L1C_WEIGHT_INDEP_REL_TOL": L1C_WEIGHT_INDEP_REL_TOL,
            "L2A_ANTIPERSEV_MARGIN_FLOOR": L2A_ANTIPERSEV_MARGIN_FLOOR,
            "L2B_REPLAY_MARGIN_TOL": L2B_REPLAY_MARGIN_TOL,
            "R1_DENSITY_DISCRIM_FLOOR": R1_DENSITY_DISCRIM_FLOOR,
            "R2_MARGIN_FLOOR": R2_MARGIN_FLOOR,
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
        print(f"  seed={seed} margin_on={s['margin_on']:.3f} margin_off={s['margin_off']:.3f} "
              f"prop={s['margin_propagation_delta']:.3f} wi_rel={s['margin_weight_indep_rel']:.3f} "
              f"antipersev={s['antipersev_margin']:.3f} replay={s['replay_margin_shift']:.3f} "
              f"r1={s['r1_density_gap']:.2f} r2_margin={s['r2_margin']:.3f} "
              f"(pref_on={s['pref_a_on']:.2f})", flush=True)

    ev = evaluate(per_seed)
    outcome = ev["outcome"]
    elapsed = time.time() - t0

    print(f"[{EXPERIMENT_TYPE}] label={ev['interpretation']['label']} "
          f"direction={ev['evidence_direction']}")
    agg = ev["aggregates"]
    print(f"  median_margin_on={agg['median_margin_on']:.3f} margin_off={agg['median_margin_off']:.3f} "
          f"prop={agg['median_margin_propagation_delta']:.3f} "
          f"wi_rel={agg['median_margin_weight_indep_rel']:.3f} "
          f"antipersev={agg['median_antipersev_margin']:.3f}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    full_config = {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "horizon": HORIZON, "da_signal": DA_SIGNAL, "da_allocation_scale": DA_ALLOCATION_SCALE,
        "da_jitter_radius": DA_JITTER_RADIUS, "n_reward_a": N_REWARD_A, "n_sparse_b": N_SPARSE_B,
        "region_sep": REGION_SEP, "visit_jitter": VISIT_JITTER, "n_trials": N_TRIALS,
        "k_candidates": K_CANDIDATES, "target_jitter": TARGET_JITTER,
        "curiosity_weight": CURIOSITY_WEIGHT, "familiarity_ema_alpha": FAMILIARITY_EMA_ALPHA,
        "familiarity_bandwidth": FAMILIARITY_BANDWIDTH, "n_fam_rounds": N_FAM_ROUNDS,
        "ctrl_curiosity_weight": CTRL_CURIOSITY_WEIGHT, "seeds": list(seeds),
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "supersedes": SUPERSEDES,
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
        "substrate": "SD-025",
        "notes": (
            "SD-025 curiosity-drive selection-bias validation (DRIVE MECHANISM only), "
            "CONTINUOUS CEM score-margin re-operationalization of V3-EXQ-767 (supersedes it). "
            "767's binary argmin pref_dense saturated to 1.0 with zero cross-seed variance "
            "(vacuous_pass / measurement_degeneracy, failure_autopsy_767-768-cluster_2026-07-16 "
            "confirmed, applied REE_assembly e40362ebb7). This iteration makes the LOAD-BEARING "
            "statistic the continuous CEM score-margin mean_trials[min(sparse score) - "
            "min(dense score)] -- already computed inside _pref_dense and previously discarded. "
            "The margin's SIGN is exactly the old binary decision; its MAGNITUDE reflects drive "
            "strength and is variance-bearing (scales with w*(density gap), which varies 58-80 "
            "across seeds). OFF counterbalances to a clean ~0 margin null. A criteria_non_degenerate "
            "guard requires margin_on to VARY across seeds so this cannot re-flag vacuous_pass. "
            "Tests that the curiosity term PROPAGATES into hippocampal CEM elite selection (a "
            "graded margin) toward higher representational density (the propagation MECH-111's "
            "broken broadcast-novelty->E3 path could not achieve, EXQ-141b/590a), reads density "
            "WEIGHT-INDEPENDENTLY (benefit value never enters ARC-007-strict scoring), and is "
            "discounted by a WAKING-only familiarity EMA (anti-perseveration; MECH-094 replay "
            "control). NOT the env-constrained full ARC-057 approach-emergence claim. No training "
            "(non-parametric RBF + read + EMA); phased training N/A. A PASS is drive-mechanism "
            "readiness routed through /failure-autopsy; it does NOT itself promote ARC-057."
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
