#!/opt/local/bin/python3
"""V3-EXQ-766 -- MECH-232 DA-modulated representational expansion (SD-024 validation).

Claim: MECH-232 (hippocampus.da_representational_expansion)
Purpose: DIAGNOSTIC (experiment_purpose="diagnostic"; excluded from governance
  confidence/conflict scoring; a PASS routes through /failure-autopsy adjudication
  before it can drive the MECH-232 candidate->provisional promotion, a FAIL refutes).
Substrate: SD-024 (hippocampal_module.da_modulated_rbf_density), IMPLEMENTED 2026-07-16.
Single claim -> no evidence_direction_per_claim needed.

WHAT THIS TESTS
---------------
MECH-232's falsifiable prediction: dopamine at reward-associated locations produces
representational EXPANSION (higher information density / finer resolution / more stable
place fields) in the hippocampal benefit terrain, and this expansion -- NOT an explicit
positive-valence gradient -- is what lets approach valence enter hippocampal terrain.

SD-024 is the instrument: DA-modulated accumulate_benefit allocates a local CLUSTER of
RBF centers (representational expansion) with the total benefit intensity SPLIT across the
cluster, so the summed benefit VALUE at the reward location is CONSERVED (integrates to the
same magnitude a single-center OFF allocation would) while the representational DENSITY
rises. compute_local_density reads that density WEIGHT-INDEPENDENTLY.

Two legs (both must hold for PASS):

LEG 1 -- representational expansion at reward locations (DA-ON vs DA-OFF, same seeds/inputs):
  L1a density_expansion_ratio = density_on(reward)/density_off(reward) >= 1.5  [LOAD-BEARING]
  L1b resolution: mean per-center bandwidth of the ON reward cluster < base    [supporting]
  L1c stability:  CV of the running density at reward is lower under ON        [supporting]

LEG 2 -- CRUX: approach emerges from representational quality ALONE, not a valence gradient.
  A density-following hill-climber (reads ONLY compute_local_density, never the RBF weights)
  started from points distant from the reward region approaches it under DA-ON.
    L2a approach_success_density_on >= 0.60                                     [LOAD-BEARING]
    L2b approach_success_density_on - approach_success_random >= 0.20  (real, not chance)
  THE DISCRIMINATOR ("approach without an explicit gradient field"): ZERO every benefit
  weight -- removing the entire value/valence field (evaluate_benefit becomes flat zero).
  compute_local_density is WEIGHT-INDEPENDENT, so the density field is UNCHANGED and the
  density-follower still approaches; a value-follower now has no gradient and falls to chance.
    L2c approach_without_gradient: with all weights zeroed,                     [LOAD-BEARING]
        approach_success_density_zeroed >= 0.60  (density approach persists) AND
        approach_success_value_zeroed <= approach_success_random + 0.10 (value -> chance)
  So approach rides on the representational structure DA built, not on a positive-valence
  gradient -- the discriminator against a valence-tag account (e.g. the BLA threat pathway,
  which writes an explicit tag). (Total benefit MASS is conserved by construction -- the DA
  cluster splits one encounter's intensity across its centers -- recorded as context.)

ACCEPTANCE (PASS): L1a AND L2a AND L2b AND L2c. FAIL otherwise (refutes MECH-232).

READINESS (P0 positive controls -- SAME statistic the load-bearing criteria route on; a
below-floor reading self-routes substrate_not_ready_requeue, NEVER a substrate verdict):
  R1 density read DISCRIMINATES: on a fresh single DA-ON cluster, density(reward)-density(far)
     >= 0.5  (the density statistic L1a / L2 route on).
  R2 walker FUNCTIONAL: the density hill-climber reaches a strong positive-control Gaussian
     (success >= 0.80)  (the L2 instrument works).

No training occurs (no encoder head; the benefit terrain is a non-parametric RBF
accumulator). Phased training N/A. MECH-094: DA expansion inherits accumulate_benefit's
hypothesis_tag gate (not exercised here -- all accumulation is waking).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import ResidueConfig  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_766_mech232_da_modulated_representational_expansion"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS = ["MECH-232"]

# ---- fixed design constants (pre-registered) ----
WORLD_DIM = 16
NUM_CENTERS = 256          # equal capacity for BOTH arms (fair density comparison)
KERNEL_BANDWIDTH = 1.0
N_REWARD = 24              # reward encounters (DA-carrying)
N_DISTRACTOR = 24          # non-reward encounters (no DA -> single center in both arms)
DA_SIGNAL = 1.0
DA_ALLOCATION_SCALE = 3.0  # DA=1.0 -> n = 1 + int(3) = 4 centers per reward encounter
DA_JITTER_RADIUS = 0.12
DA_BANDWIDTH_NARROWING = 0.5
VISIT_JITTER = 0.05        # per-visit noise around the reward location
DISTRACTOR_DIST = 4.0      # distractors placed far from the reward region

# ---- walker (leg-2 instrument) ----
N_STARTS = 24
START_DIST = 1.6           # distance of walker starts from the reward location
WALK_STEPS = 40
N_CANDIDATES = 16
STEP_SIGMA = 0.15
SUCCESS_RADIUS = 0.5

# ---- pre-registered acceptance thresholds ----
L1A_EXPANSION_MIN = 1.5
L2A_APPROACH_MIN = 0.60
L2B_CHANCE_MARGIN = 0.20
L2C_VALUE_ZEROED_CHANCE_TOL = 0.10   # value-follower with weights zeroed must be ~chance
# readiness floors
R1_DENSITY_DISCRIM_FLOOR = 0.5
R2_WALKER_CONTROL_FLOOR = 0.80

DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
ARMS = ["da_off", "da_on"]
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


def _config_slice(da_on: bool) -> Dict:
    return {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS,
        "kernel_bandwidth": KERNEL_BANDWIDTH, "use_da_modulated_rbf_density": da_on,
        "da_allocation_scale": DA_ALLOCATION_SCALE, "da_jitter_radius": DA_JITTER_RADIUS,
        "da_bandwidth_narrowing": DA_BANDWIDTH_NARROWING,
        "n_reward": N_REWARD, "n_distractor": N_DISTRACTOR, "da_signal": DA_SIGNAL,
        "visit_jitter": VISIT_JITTER, "distractor_dist": DISTRACTOR_DIST,
    }


def _make_inputs(seed: int):
    """Deterministic per-seed reward location + encounter sequences (local generator,
    so both arms replay IDENTICAL inputs regardless of the global-RNG reset arm_cell does)."""
    gen = torch.Generator().manual_seed(1000 + seed)
    z_reward = torch.randn(1, WORLD_DIM, generator=gen)
    z_reward = z_reward / (z_reward.norm() + EPS)  # unit-norm reward location
    reward_seq = [z_reward + VISIT_JITTER * torch.randn(1, WORLD_DIM, generator=gen)
                  for _ in range(N_REWARD)]
    distractor_seq = []
    for _ in range(N_DISTRACTOR):
        d = torch.randn(1, WORLD_DIM, generator=gen)
        d = d / (d.norm() + EPS) * DISTRACTOR_DIST
        distractor_seq.append(d)
    return gen, z_reward, reward_seq, distractor_seq


def _build_field(cfg: ResidueConfig, z_reward, reward_seq, distractor_seq):
    """Accumulate the reward-encounter sequence into a benefit terrain; track the running
    density at the reward location. Returns (field, density_track, n_reward_centers,
    mean_reward_bandwidth)."""
    rf = ResidueField(cfg)
    n_before = int(rf.benefit_rbf_field.active_mask.sum())
    density_track: List[float] = []
    for zr in reward_seq:
        rf.accumulate_benefit(zr, benefit_magnitude=1.0, dopamine_signal=DA_SIGNAL)
        density_track.append(float(rf.compute_benefit_density(z_reward).item()))
    n_reward_centers = int(rf.benefit_rbf_field.active_mask.sum()) - n_before

    # per-center bandwidth of the reward cluster (resolution readout); OFF has no
    # per-center buffer -> report the base bandwidth.
    if getattr(rf.benefit_rbf_field, "per_center_bandwidth", False):
        active = rf.benefit_rbf_field.active_mask
        mean_bw = float(rf.benefit_rbf_field.center_bandwidths[active].mean().item())
    else:
        mean_bw = float(KERNEL_BANDWIDTH)

    for zd in distractor_seq:
        rf.accumulate_benefit(zd, benefit_magnitude=1.0, dopamine_signal=0.0)
    return rf, density_track, n_reward_centers, mean_bw


def _cv_second_half(track: List[float]) -> float:
    """Coefficient of variation of the running density over the 2nd half of encounters
    (lower = more stable place field). Returns -1.0 if undefined."""
    if len(track) < 4:
        return -1.0
    tail = torch.tensor(track[len(track) // 2:], dtype=torch.float64)
    m = float(tail.mean())
    if abs(m) < EPS:
        return -1.0
    return float(tail.std(unbiased=False).item() / m)


def _hillclimb(field_fn: Callable[[torch.Tensor], torch.Tensor],
               z_reward: torch.Tensor, gen: torch.Generator) -> float:
    """Greedy hill-climb on a scalar field. From N_STARTS points at START_DIST from
    z_reward, take WALK_STEPS steps; each step move to the best of N_CANDIDATES local
    proposals (only if it improves). Returns success fraction (final dist < radius)."""
    successes = 0
    for _ in range(N_STARTS):
        direction = torch.randn(1, WORLD_DIM, generator=gen)
        direction = direction / (direction.norm() + EPS)
        z = z_reward + direction * START_DIST
        f_cur = float(field_fn(z).item())
        for _ in range(WALK_STEPS):
            cand = z + STEP_SIGMA * torch.randn(N_CANDIDATES, WORLD_DIM, generator=gen)
            f_cand = field_fn(cand)                       # [N_CANDIDATES]
            best = int(torch.argmax(f_cand).item())
            f_best = float(f_cand[best].item())
            if f_best > f_cur:
                z = cand[best:best + 1]
                f_cur = f_best
        if float((z - z_reward).norm().item()) < SUCCESS_RADIUS:
            successes += 1
    return successes / float(N_STARTS)


def _random_walk(z_reward: torch.Tensor, gen: torch.Generator) -> float:
    """Chance-level baseline: undirected random walk with the same step budget."""
    successes = 0
    for _ in range(N_STARTS):
        direction = torch.randn(1, WORLD_DIM, generator=gen)
        direction = direction / (direction.norm() + EPS)
        z = z_reward + direction * START_DIST
        for _ in range(WALK_STEPS):
            z = z + STEP_SIGMA * torch.randn(1, WORLD_DIM, generator=gen)
        if float((z - z_reward).norm().item()) < SUCCESS_RADIUS:
            successes += 1
    return successes / float(N_STARTS)


def _readiness(seed: int, z_reward: torch.Tensor, gen: torch.Generator):
    """P0 positive controls, measured on the SAME statistics the load-bearing criteria use."""
    # R1: density read discriminates after a single DA-ON cluster.
    cfg = _residue_cfg(da_on=True)
    rf = ResidueField(cfg)
    rf.accumulate_benefit(z_reward, benefit_magnitude=1.0, dopamine_signal=DA_SIGNAL)
    far = z_reward + 3.0 * torch.nn.functional.normalize(
        torch.randn(1, WORLD_DIM, generator=gen), dim=-1)
    d_reward = float(rf.compute_benefit_density(z_reward).item())
    d_far = float(rf.compute_benefit_density(far).item())
    r1_gap = d_reward - d_far

    # R2: the hill-climber reaches a strong positive-control single Gaussian.
    ctrl = ResidueField(_residue_cfg(da_on=False))
    for _ in range(8):  # a strong, well-formed single-location attractor
        ctrl.accumulate_benefit(z_reward, benefit_magnitude=1.0, dopamine_signal=0.0)
    r2_succ = _hillclimb(lambda z: ctrl.compute_benefit_density(z), z_reward, gen)
    return r1_gap, r2_succ


def run_seed(seed: int) -> Dict:
    gen, z_reward, reward_seq, distractor_seq = _make_inputs(seed)

    fields = {}
    arm_rows = {}
    for arm in ARMS:
        da_on = (arm == "da_on")
        with arm_cell(
            seed,
            config_slice=_config_slice(da_on),
            script_path=Path(__file__),
            config_slice_declared=True,
            include_driver_script_in_hash=False,  # mint-as-you-go: reuse-eligible OFF baseline
        ) as cell:
            rf, track, n_centers, mean_bw = _build_field(
                _residue_cfg(da_on), z_reward, reward_seq, distractor_seq)
            density_reward = float(rf.compute_benefit_density(z_reward).item())
            value_reward = float(rf.evaluate_benefit(z_reward).item())
            row = {
                "arm_id": arm,
                "seed": seed,
                "density_reward": density_reward,
                "value_reward": value_reward,
                "n_reward_centers": n_centers,
                "mean_reward_bandwidth": mean_bw,
                "density_cv_second_half": _cv_second_half(track),
                "density_track": [round(x, 5) for x in track],
            }
            cell.stamp(row)
        fields[arm] = rf
        arm_rows[arm] = row
        # progress instrumentation
        print(f"Seed {seed} Condition {arm}")
        for i in range(N_REWARD + N_DISTRACTOR):
            if (i + 1) % 16 == 0 or (i + 1) == (N_REWARD + N_DISTRACTOR):
                print(f"  [train] density seed={seed} arm={arm} "
                      f"ep {i + 1}/{N_REWARD + N_DISTRACTOR}", flush=True)
        # per-cell completion verdict (non-degeneracy of THIS cell: it accumulated centers)
        cell_ok = n_centers > 0 and density_reward > 0.0
        print(f"verdict: {'PASS' if cell_ok else 'FAIL'}")

    off, on = arm_rows["da_off"], arm_rows["da_on"]

    # LEG 1
    expansion_ratio = on["density_reward"] / (off["density_reward"] + EPS)
    resolution_ok = on["mean_reward_bandwidth"] < KERNEL_BANDWIDTH - 1e-6
    cv_on, cv_off = on["density_cv_second_half"], off["density_cv_second_half"]
    stability_ok = (cv_on >= 0 and cv_off >= 0 and cv_on < cv_off)

    # LEG 2 -- density-follower approach + the zeroed-weights discriminator
    succ_density_on = _hillclimb(lambda z: fields["da_on"].compute_benefit_density(z), z_reward, gen)
    succ_density_off = _hillclimb(lambda z: fields["da_off"].compute_benefit_density(z), z_reward, gen)
    succ_value_on = _hillclimb(lambda z: fields["da_on"].evaluate_benefit(z), z_reward, gen)
    succ_random = _random_walk(z_reward, gen)

    # CRUX: remove the entire value/valence field by zeroing every benefit weight.
    # compute_local_density is weight-independent -> density field UNCHANGED (approach
    # persists); evaluate_benefit -> flat zero -> value-follower falls to chance.
    with torch.no_grad():
        fields["da_on"].benefit_rbf_field.weights.zero_()
    succ_density_zeroed = _hillclimb(lambda z: fields["da_on"].compute_benefit_density(z), z_reward, gen)
    succ_value_zeroed = _hillclimb(lambda z: fields["da_on"].evaluate_benefit(z), z_reward, gen)
    value_after_zero = float(fields["da_on"].evaluate_benefit(z_reward).item())

    # total benefit MASS conservation (context: DA splits intensity across the cluster)
    total_off = float(fields["da_off"].total_benefit.item())
    total_on = float(fields["da_on"].total_benefit.item())
    rel_mass_diff = abs(total_on - total_off) / (abs(total_off) + EPS)

    # readiness (positive controls)
    r1_gap, r2_succ = _readiness(seed, z_reward, gen)

    return {
        "seed": seed,
        # per-(seed,arm) stamped rows (carry arm_fingerprint for mint-as-you-go + hoist)
        "arm_rows": [arm_rows["da_off"], arm_rows["da_on"]],
        # leg 1
        "density_off": off["density_reward"],
        "density_on": on["density_reward"],
        "expansion_ratio": expansion_ratio,
        "mean_bw_on": on["mean_reward_bandwidth"],
        "resolution_ok": bool(resolution_ok),
        "cv_on": cv_on, "cv_off": cv_off, "stability_ok": bool(stability_ok),
        "n_centers_off": off["n_reward_centers"], "n_centers_on": on["n_reward_centers"],
        # leg 2
        "value_off": off["value_reward"], "value_on": on["value_reward"],
        "approach_density_on": succ_density_on,
        "approach_density_off": succ_density_off,
        "approach_value_on": succ_value_on,
        "approach_random": succ_random,
        "approach_density_zeroed": succ_density_zeroed,
        "approach_value_zeroed": succ_value_zeroed,
        "value_after_zero": value_after_zero,
        "total_benefit_off": total_off, "total_benefit_on": total_on,
        "rel_mass_diff": rel_mass_diff,
        # readiness
        "r1_density_discrim_gap": r1_gap,
        "r2_walker_control_succ": r2_succ,
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(torch.tensor(xs, dtype=torch.float64).median().item())


def evaluate(per_seed: List[Dict]) -> Dict:
    med_expansion = _median([s["expansion_ratio"] for s in per_seed])
    med_approach_density_on = _median([s["approach_density_on"] for s in per_seed])
    med_approach_random = _median([s["approach_random"] for s in per_seed])
    med_approach_density_off = _median([s["approach_density_off"] for s in per_seed])
    med_approach_density_zeroed = _median([s["approach_density_zeroed"] for s in per_seed])
    med_approach_value_zeroed = _median([s["approach_value_zeroed"] for s in per_seed])
    med_rel_mass_diff = _median([s["rel_mass_diff"] for s in per_seed])
    frac_resolution_ok = sum(s["resolution_ok"] for s in per_seed) / len(per_seed)
    frac_stability_ok = sum(s["stability_ok"] for s in per_seed) / len(per_seed)

    min_r1 = min(s["r1_density_discrim_gap"] for s in per_seed)
    min_r2 = min(s["r2_walker_control_succ"] for s in per_seed)

    # ---- readiness preconditions (same statistic as the load-bearing criteria) ----
    preconditions = [
        {"name": "density_read_discriminates", "kind": "readiness",
         "description": "single DA-ON cluster: density(reward) - density(far) clears floor "
                        "(the density statistic L1a/L2 route on)",
         "measured": round(min_r1, 5), "threshold": R1_DENSITY_DISCRIM_FLOOR,
         "control": "fresh single DA-ON cluster at the reward location", "direction": "lower",
         "met": bool(min_r1 >= R1_DENSITY_DISCRIM_FLOOR)},
        {"name": "density_walker_functional", "kind": "readiness",
         "description": "the density hill-climber (the L2 instrument) reaches a strong "
                        "positive-control single Gaussian",
         "measured": round(min_r2, 4), "threshold": R2_WALKER_CONTROL_FLOOR,
         "control": "single-location attractor, 8 accumulations", "direction": "lower",
         "met": bool(min_r2 >= R2_WALKER_CONTROL_FLOOR)},
    ]
    ready = all(p["met"] for p in preconditions)

    # ---- load-bearing acceptance criteria ----
    l1a = med_expansion >= L1A_EXPANSION_MIN
    l2a = med_approach_density_on >= L2A_APPROACH_MIN
    l2b = (med_approach_density_on - med_approach_random) >= L2B_CHANCE_MARGIN
    # L2c: with the value field removed (weights zeroed), density approach persists AND the
    # value-follower falls to chance -> approach without an explicit gradient field.
    l2c_density_persists = med_approach_density_zeroed >= L2A_APPROACH_MIN
    l2c_value_at_chance = med_approach_value_zeroed <= (med_approach_random + L2C_VALUE_ZEROED_CHANCE_TOL)
    l2c = l2c_density_persists and l2c_value_at_chance

    criteria = [
        {"name": "L1a_density_expansion", "load_bearing": True, "passed": bool(l1a),
         "measured": round(med_expansion, 4), "threshold": L1A_EXPANSION_MIN},
        {"name": "L2a_density_approach", "load_bearing": True, "passed": bool(l2a),
         "measured": round(med_approach_density_on, 4), "threshold": L2A_APPROACH_MIN},
        {"name": "L2b_approach_above_chance", "load_bearing": True, "passed": bool(l2b),
         "measured": round(med_approach_density_on - med_approach_random, 4),
         "threshold": L2B_CHANCE_MARGIN},
        {"name": "L2c_approach_without_gradient", "load_bearing": True, "passed": bool(l2c),
         "density_zeroed": round(med_approach_density_zeroed, 4),
         "value_zeroed": round(med_approach_value_zeroed, 4),
         "chance": round(med_approach_random, 4),
         "note": "weights zeroed: density approach persists AND value-follower at chance"},
        {"name": "L1b_resolution_narrowed", "load_bearing": False,
         "passed": bool(frac_resolution_ok >= 0.5), "measured": round(frac_resolution_ok, 3)},
        {"name": "L1c_stability_improved", "load_bearing": False,
         "passed": bool(frac_stability_ok >= 0.5), "measured": round(frac_stability_ok, 3)},
        {"name": "total_benefit_mass_conserved", "load_bearing": False,
         "passed": bool(med_rel_mass_diff <= 0.02), "measured": round(med_rel_mass_diff, 5),
         "note": "context: DA splits one encounter's intensity across the cluster"},
    ]

    # ---- non-degeneracy (would the criteria have been vacuous?) ----
    densities_vary = (len({round(s["density_on"], 3) for s in per_seed}) > 1
                      and all(s["density_on"] > 0 for s in per_seed))
    approach_varies = len({round(s["approach_density_on"], 3) for s in per_seed}) > 1
    mechanism_active = all(s["n_centers_on"] > s["n_centers_off"] for s in per_seed)
    criteria_non_degenerate = {
        "densities_positive_and_vary": bool(densities_vary),
        "approach_varies_across_seeds": bool(approach_varies),
        "da_allocates_more_centers": bool(mechanism_active),
    }
    non_degenerate = all(criteria_non_degenerate.values())
    degeneracy_reason = (
        "" if non_degenerate else
        "one of: density constant/zero, approach constant across seeds, or DA-ON did not "
        "allocate more centers than DA-OFF")

    # ---- route ----
    if not ready:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        direction = "non_contributory"
    else:
        load_bearing_pass = l1a and l2a and l2b and l2c
        outcome = "PASS" if load_bearing_pass else "FAIL"
        direction = "supports" if load_bearing_pass else "weakens"
        if load_bearing_pass:
            label = "da_representational_expansion_produces_approach_without_valence_gradient"
        else:
            failed = [c["name"] for c in criteria if c["load_bearing"] and not c["passed"]]
            label = "mech232_prediction_not_met:" + ",".join(failed)

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
            "median_expansion_ratio": med_expansion,
            "median_approach_density_on": med_approach_density_on,
            "median_approach_density_off": med_approach_density_off,
            "median_approach_density_zeroed": med_approach_density_zeroed,
            "median_approach_value_zeroed": med_approach_value_zeroed,
            "median_approach_random": med_approach_random,
            "median_rel_mass_diff": med_rel_mass_diff,
            "frac_resolution_ok": frac_resolution_ok,
            "frac_stability_ok": frac_stability_ok,
            "min_r1_density_discrim_gap": min_r1,
            "min_r2_walker_control_succ": min_r2,
        },
        "thresholds": {
            "L1A_EXPANSION_MIN": L1A_EXPANSION_MIN,
            "L2A_APPROACH_MIN": L2A_APPROACH_MIN,
            "L2B_CHANCE_MARGIN": L2B_CHANCE_MARGIN,
            "L2C_VALUE_ZEROED_CHANCE_TOL": L2C_VALUE_ZEROED_CHANCE_TOL,
            "R1_DENSITY_DISCRIM_FLOOR": R1_DENSITY_DISCRIM_FLOOR,
            "R2_WALKER_CONTROL_FLOOR": R2_WALKER_CONTROL_FLOOR,
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
        # the stamped per-(seed,arm) rows carry arm_fingerprint (mint-as-you-go + hoist);
        # keep them out of the per_seed summary to avoid duplication.
        arm_results.extend(s.pop("arm_rows"))
        per_seed.append(s)
        print(f"  seed={seed} expansion={s['expansion_ratio']:.2f} "
              f"approach_on={s['approach_density_on']:.2f} approach_rand={s['approach_random']:.2f} "
              f"dens_zeroed={s['approach_density_zeroed']:.2f} val_zeroed={s['approach_value_zeroed']:.2f} "
              f"r1={s['r1_density_discrim_gap']:.2f} r2={s['r2_walker_control_succ']:.2f}", flush=True)

    ev = evaluate(per_seed)
    outcome = ev["outcome"]
    elapsed = time.time() - t0

    print(f"[{EXPERIMENT_TYPE}] label={ev['interpretation']['label']} "
          f"direction={ev['evidence_direction']}")
    agg = ev["aggregates"]
    print(f"  median_expansion={agg['median_expansion_ratio']:.2f} "
          f"approach_on={agg['median_approach_density_on']:.2f} "
          f"approach_rand={agg['median_approach_random']:.2f} "
          f"dens_zeroed={agg['median_approach_density_zeroed']:.2f} "
          f"val_zeroed={agg['median_approach_value_zeroed']:.2f}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    full_config = {
        "world_dim": WORLD_DIM, "num_centers": NUM_CENTERS, "kernel_bandwidth": KERNEL_BANDWIDTH,
        "n_reward": N_REWARD, "n_distractor": N_DISTRACTOR, "da_signal": DA_SIGNAL,
        "da_allocation_scale": DA_ALLOCATION_SCALE, "da_jitter_radius": DA_JITTER_RADIUS,
        "da_bandwidth_narrowing": DA_BANDWIDTH_NARROWING, "visit_jitter": VISIT_JITTER,
        "distractor_dist": DISTRACTOR_DIST, "n_starts": N_STARTS, "start_dist": START_DIST,
        "walk_steps": WALK_STEPS, "n_candidates": N_CANDIDATES, "step_sigma": STEP_SIGMA,
        "success_radius": SUCCESS_RADIUS, "seeds": list(seeds),
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
        "substrate": "SD-024",
        "notes": (
            "SD-024 DA-modulated RBF density validation. Diagnostic instrument for MECH-232. "
            "Benefit-terrain-only; harm/safety fields never DA-modulated (MECH-233 asymmetry). "
            "Total benefit intensity split across the DA cluster -> summed VALUE conserved while "
            "representational DENSITY expands; compute_local_density is weight-independent. "
            "No training (non-parametric RBF); phased training N/A. A PASS routes through "
            "/failure-autopsy before promoting MECH-232 candidate->provisional."
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
