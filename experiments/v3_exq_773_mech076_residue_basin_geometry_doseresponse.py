#!/opt/local/bin/python3
"""V3-EXQ-773 -- MECH-076 residue-as-terrain-deformation: basin-geometry dose-response.

Claim: MECH-076 (residue.hippocampal_map_structural_deformation) -- residue (the
ARC-013 curvature field phi(z_world)) is instantiated as GEOMETRIC DEFORMATION of the
map terrain (deep attractor basins), acting as trajectory-level pressure THROUGH the
basin geometry, NOT as a separate inhibitory / scoring filter layered on perception.
Status: candidate (v3_pending=false). depends_on ARC-013, MECH-056, MECH-073.
Purpose: evidence (confirming/falsifying experiment). Single claim -> no per-claim direction.

What is DISTINCT here (vs neighbouring evidence already on record)
-----------------------------------------------------------------
  * INV-053 (V3-EXQ-249/268/406/434) showed a deep basin produces lock-in as a BINARY
    attractor-present/absent depression state. This is NOT that: it measures a GRADED
    dose-response, not a single deep-basin regime.
  * ARC-013 / Q-079 (V3-EXQ-697) tested residue's DECISION-SELECTION separability
    (irreducible to salience + uncertainty). This is NOT that: it does not test
    reducibility; it tests whether influence is mediated by basin GEOMETRY.
  * The load-bearing MECH-076 sub-claim isolated here: residue's behavioural influence
    is GRADED TERRAIN GEOMETRY -- basin depth is a continuous causal dose whose
    behavioural effect (trajectory repulsion under the substrate's terrain-only elite
    selection) (C1) increases MONOTONICALLY with depth, and (C2) is MEDIATED BY THE
    FIELD GEOMETRY: the RBF spatial gradient exerts repulsion AT A DISTANCE (candidates
    that pass ADJACENT to the basin -- never touching its peak -- still lose selection
    mass, and that loss grows with depth). The alternative (a discrete lookup-gate that
    only penalises candidates landing ON the peak) predicts C2 null: only peak-band
    candidates lose mass, adjacent candidates are untouched. So the result CAN come out
    either way -- if the terrain-cost -> elite-selection map behaves as a de-facto
    peak-only cutoff, C2 fails and the reading is does_not_support.

SCOPE HONESTY (stated up front)
-------------------------------
MECH-076 lists FIVE structural-deformation mechanisms. The V3 substrate expresses ONE:
deep attractor basins as RBF terrain deformation (ResidueField over z_world). It does
NOT implement: map-metric over-representation (the z_world encoder is fixed -- residue
is an overlay field, not a metric distortion), dentate-gyrus neurogenesis / pattern-
separation modulation, CA3 dendritic retraction, or palimpsest reconsolidation traces.
So this probe tests the BASIN-GEOMETRY LEG only; evidence_class = "mechanistic_partial"
(partial mechanistic support for that one sub-mechanism, never the full claim).

WALL-INDEPENDENT by construction
--------------------------------
The DV is a passive terrain readout + terrain-only elite selection over a fixed candidate
bank -- NO planning, NO action-commitment, NO goal-directed policy, NO training. So the
verdict is independent of the V3 competence wall (precedent: V3-EXQ-760/455/447/448).

No training occurs
------------------
The encoder is an UNTRAINED, frozen REEAgent used purely as a deterministic z_world
feature map (torch.no_grad forward passes) to GROUND z_world in real substrate latents
(answers the synthetic_signals_only why-now flag -- z* and every candidate trajectory
are real z_world states from random walks, not hand-crafted vectors). The residue terrain
is non-parametric (RBFLayer.add_residue -- no gradients). Phased training is N/A (no head
is trained). alpha_world (a training-loss weight) does not affect a no-grad forward pass.

Design (grid = seed x depth-arm)
--------------------------------
Per seed (RNG reset once):
  1. Frozen untrained REEAgent + CausalGridWorldV2. Random walks collect a POOL of real
     z_world states and a BANK of real z_world trajectories (contiguous HORIZON windows).
  2. z* = medoid of the pool (a central real z_world location -- the aversive site).
  3. bw = median nearest-neighbour distance of the pool (the natural neighbourhood scale);
     the basin bandwidth is set to bw so band radii are commensurate with real z_world
     spacing regardless of the encoder's absolute scale.
  4. bw = median(d_min)/2 so the median candidate sits at ~2*bw. Candidates are binned by
     PERCENTILE of the closest-approach d_min distribution (guarantees populated bands):
     PEAK (closest PEAK_PCT), CAPTURE (CAP_LO_PCT..CAP_HI_PCT, around the median ~2*bw --
     adjacent, never touching the peak: phi ~= 13.5% of peak, gradient clearly nonzero),
     FAR (farthest, beyond FAR_PCT).
Per dose-arm (multiplier m in DOSE_MULTIPLIERS; m=0 == OFF):
  5. Depth is dosed ADAPTIVELY: weight = m * w_ref, where w_ref (per seed) is the weight
     that shifts a peak trajectory's terrain cost by ~1 selection-unit (= OFF cost std /
     mean unit-weight peak cost increment). This centres the dose-response in the
     responsive regime regardless of the encoder scale. If m>0, one accumulate(z*,
     harm_magnitude=m*w_ref) places a single RBF repeller centre at z*.
  6. Terrain cost per candidate = evaluate_trajectory(traj) (the REAL terrain-cost
     function the hippocampal module calls, module.py:_score_trajectory -> line ~838,
     ARC-007 STRICT terrain-only, no value head). Elite-selection probability
     p_i = softmax(-beta * cost_i) over the bank (the continuous form of the module's
     CEM elite refit; beta is fixed from the OFF-arm cost spread, applied identically to
     every arm). avoidance_i(D) = p_i(OFF) - p_i(D) = selection mass the basin pulls off
     candidate i. Band avoidance = sum of avoidance over the band's candidates.

DV / metrics (per seed)
-----------------------
  peak_avoidance(D)       = PEAK-band selection mass lost vs OFF (unconfounded: PEAK
                            unambiguously loses mass to a repeller). C1 metric.
  capture_vs_far_shift(D) = drop in CAPTURE's selection share RELATIVE TO FAR vs OFF.
                            Both bands are off-peak (never touch z*), so their relative
                            shift removes the softmax mass-redistribution artifact that a
                            repeller injects into every non-peak band. Positive = the RBF
                            gradient reaches the capture distance and disadvantages the
                            closer (capture) candidates more than the far ones. C2 metric.
  basin_peak_depth(D)     = evaluate(z*) - evaluate_off(z*)  (geometry readout).
  field_contrast_capture(D) = |evaluate(z*) - evaluate(z* + R_CAPTURE_MID*bw*u_hat)| /
                            basin_peak_depth, mean over directions (RELATIVE contrast,
                            scale-free ~= 0.86 by RBF construction). Readiness positive
                            control: confirms the geometry reaches the capture band.

Readiness / non-vacuity (self-route non_contributory, NOT a refutation)
-----------------------------------------------------------------------
  P1 geometry_reaches_capture_band: mean field_contrast_capture(deepest D) >= FIELD_CONTRAST_FLOOR.
     (Same statistic C2 depends on: if the basin's own field does not vary across the
     capture band, C2 cannot be a real test -- the substrate is not ready, not refuted.)
  Non-vacuity: every band (PEAK/CAPTURE/FAR) has >= BAND_MIN_COUNT candidates in
     >= SEED_PASS_N seeds. If unmet -> non_contributory (bank geometry too sparse to test).

Pre-registered PASS criteria (thresholds fixed here; both load-bearing)
----------------------------------------------------------------------
On the 6-point adaptively-dosed curve a clean saturating-monotone response gives Spearman
>= ~0.94; a flat / noisy / inverted (discrete-gate or no-effect) curve fails 0.9. Each gate
also demands a real effect floor so a monotone-but-negligible curve cannot pass.
  C1 (load_bearing): per-seed Spearman(DOSE_MULTIPLIERS, peak_avoidance) >= SPEARMAN_MONO
     AND peak_avoidance(deepest) >= PEAK_EFFECT_FLOOR in >= SEED_PASS_N seeds
     (dose-response: deeper basin -> more peak repulsion).
  C2 (load_bearing): per-seed Spearman(DOSE_MULTIPLIERS, capture_vs_far_shift) >= SPEARMAN_MONO
     AND capture_vs_far_shift(deepest) >= CAPTURE_SHIFT_FLOOR in >= SEED_PASS_N seeds
     (geometry-mediated, at-a-distance: off-peak CAPTURE candidates lose selection share
     relative to FAR, graded in dose; a discrete peak-only gate leaves this ~ 0).
Outcome routing:
  readiness/non-vacuity unmet (majority) -> FAIL / non_contributory / substrate_not_ready_requeue.
  C1 and C2 -> PASS / supports (basin-geometry leg of MECH-076 confirmed; partial support).
  readiness met but C1 or C2 fail -> FAIL / does_not_support
     (graded at-a-distance geometry mediation absent -> discrete-gate reading;
      route to /failure-autopsy).

experiment_purpose = "evidence"
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

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig, ResidueConfig  # noqa: E402
from ree_core.residue.field import ResidueField  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_773_mech076_residue_basin_geometry_doseresponse"
CLAIM_IDS = ["MECH-076"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EVIDENCE_CLASS = "mechanistic_partial"  # basin-geometry leg only (see SCOPE HONESTY)

# ---- Environment (z_world grounding walks) ----
GRID_SIZE = 7
NUM_HAZARDS = 3
NUM_RESOURCES = 2
WORLD_DIM = 32

# ---- Sampling ----
POOL_STEPS = 400        # real z_world states collected per seed (medoid + bw estimate)
N_CANDIDATES = 240      # candidate trajectories in the bank per seed
HORIZON = 6             # z_world steps per candidate trajectory
N_DIRECTIONS = 16       # radial directions for the geometry readout

# ---- Residue terrain ----
NUM_BASIS = 64          # 1 active repeller centre + headroom (no ring-buffer eviction)
ACCUMULATION_RATE = 1.0  # basin weight == harm_magnitude (interpretability)
# Basin depth is dosed ADAPTIVELY as multiples of a per-seed reference weight w_ref (the
# weight that shifts a peak trajectory's terrain cost by ~1 selection-unit == std of the
# OFF-arm cost spread). This centres the dose-response in the RESPONSIVE regime regardless
# of the encoder's absolute z_world scale -- a fixed absolute depth would sit entirely in
# the saturated regime on a tightly-clustered encoder (empirically bw ~ 0.02), collapsing
# the dose-response to a single jump. Multiplier 0.0 == OFF baseline.
DOSE_MULTIPLIERS: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

# ---- Band definitions (percentiles of the closest-approach d_min distribution) ----
# Bands are percentile-based so they are ALWAYS populated; bw (= median(d_min)/2) places
# each percentile band in its intended phi regime (see build_seed_geometry).
PEAK_PCT = 0.25         # PEAK band: closest 25% of candidates (basin core)
CAP_LO_PCT = 0.45       # CAPTURE band: 45th..70th percentile of d_min (~2*bw, off-core,
CAP_HI_PCT = 0.70       #   never touching the peak; phi small, gradient nonzero)
FAR_PCT = 0.80          # FAR band: farthest 20% (essentially outside the basin)
R_CAPTURE_MID = 2.0     # readiness field-contrast probe radius in units of bw (phi ~= 13.5% peak)

# ---- Pre-registered thresholds ----
# On the 6-point adaptively-dosed curve a clean saturating-monotone response gives
# Spearman >= ~0.94 (one or two top-saturation ties), so 0.9 is met by a graded rise and
# failed by a flat / noisy / inverted (discrete-gate or no-effect) curve.
SPEARMAN_MONO = 0.9         # dose-response monotonicity gate (rho over DOSE_MULTIPLIERS)
PEAK_EFFECT_FLOOR = 0.05    # C1: peak-band selection mass moved at deepest dose >= 5%
CAPTURE_SHIFT_FLOOR = 0.01  # C2: capture-vs-far selection share shift at deepest dose >= 1%
FIELD_CONTRAST_FLOOR = 0.30  # P1 readiness: RELATIVE field contrast across capture band
                             #   (~0.86 by RBF construction; << 0.30 => geometry does not
                             #    reach the capture band -> not-ready, not a refutation)
BAND_MIN_COUNT = 8          # non-vacuity: candidates per band
SEED_PASS_N = 4             # >= 4/5 seeds

SEEDS: Tuple[int, ...] = (42, 7, 13, 99, 17)


# ------------------------------------------------------------------ helpers

def spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation (tie-aware via average ranks)."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    if vx == 0.0 or vy == 0.0:
        return float("nan")
    return cov / (vx * vy)


def softmax_neg(costs: torch.Tensor, beta: float) -> torch.Tensor:
    """Elite-selection probability p_i = softmax(-beta * cost_i) (numerically stable)."""
    logits = -beta * costs
    logits = logits - logits.max()
    ex = torch.exp(logits)
    return ex / ex.sum()


def build_agent(env: CausalGridWorldV2) -> REEAgent:
    """Untrained frozen REEAgent used only as a deterministic z_world feature map."""
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        world_dim=WORLD_DIM,
    )
    agent = REEAgent(config)
    agent.eval()
    return agent


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=NUM_RESOURCES,
        seed=seed,
        use_proxy_fields=True,
    )


def collect_zworld(env: CausalGridWorldV2, agent: REEAgent, n_steps: int,
                   action_seed: int) -> torch.Tensor:
    """Random walk (no planning). Returns [n_steps, world_dim] real z_world states."""
    gen = torch.Generator().manual_seed(action_seed)
    zs: List[torch.Tensor] = []
    agent.reset()
    _, obs_dict = env.reset()
    with torch.no_grad():
        for _ in range(n_steps):
            latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            zs.append(latent.z_world.detach().clone().reshape(-1))  # [world_dim]
            action = int(torch.randint(0, env.action_dim, (1,), generator=gen).item())
            _, _, done, _, obs_dict = env.step(torch.tensor(action))
            if done:
                agent.reset()
                _, obs_dict = env.reset()
    return torch.stack(zs, dim=0)  # [n_steps, world_dim]


def build_seed_geometry(seed: int, pool_steps: int, n_candidates: int,
                        horizon: int) -> Dict:
    """Build the real-z_world pool, candidate bank, z*, bw, and band assignments."""
    env = make_env(seed)
    agent = build_agent(env)

    # Pool for medoid + bandwidth estimate.
    pool = collect_zworld(env, agent, pool_steps, action_seed=seed * 991 + 1)  # [P, D]

    # Candidate trajectory bank: contiguous HORIZON windows from several walks.
    windows: List[torch.Tensor] = []
    w_seed = seed * 997 + 7
    guard = 0
    while len(windows) < n_candidates and guard < n_candidates * 4 + 32:
        walk = collect_zworld(env, agent, horizon * 6, action_seed=w_seed)
        w_seed += 1
        guard += 1
        for start in range(0, walk.shape[0] - horizon + 1, 2):
            windows.append(walk[start:start + horizon].clone())  # [H, D]
            if len(windows) >= n_candidates:
                break
    cand = torch.stack(windows[:n_candidates], dim=0)  # [N, H, D]

    # z* = medoid over a subsample of the pool (central real z_world location, so
    # candidate trajectories approach it from many sides).
    sub_n = min(pool.shape[0], 150)
    idx = torch.randperm(pool.shape[0])[:sub_n]
    sub = pool[idx]                                  # [S, D]
    dmat = torch.cdist(sub, sub)                     # [S, S]
    medoid_i = int(dmat.sum(dim=1).argmin().item())
    zstar = sub[medoid_i].clone()                    # [D]

    # Closest-approach distance of each candidate trajectory to z*.
    diffs = cand - zstar.reshape(1, 1, -1)           # [N, H, D]
    dstep = diffs.norm(dim=-1)                        # [N, H]
    d_min = dstep.min(dim=1).values                  # [N]

    # Basin bandwidth is set to the CLOSEST-APPROACH scale (median d_min / 2), NOT the
    # pool nearest-neighbour distance: a random trajectory's closest approach to a fixed
    # point is far larger than the global NN spacing, so anchoring bw to the NN distance
    # would leave the basin far too narrow to interact with any trajectory (empty bands).
    # With bw = median(d_min)/2 the median candidate sits at ~2*bw (phi ~= 13.5% of peak,
    # gradient clearly nonzero) -- exactly the "at a distance" regime C2 must probe.
    med_dmin = float(d_min.median().item())
    bw = med_dmin / 2.0
    if not (bw > 0.0):
        bw = 1.0

    # Bands by PERCENTILE of the d_min distribution (guarantees population by
    # construction); the bw choice above places each percentile band in the intended
    # phi regime -- PEAK (closest quartile) in the basin core, CAPTURE (around the
    # median, ~2*bw) in the small-phi/nonzero-gradient shell, FAR (farthest fifth)
    # essentially outside the basin. None of the CAPTURE candidates touch the peak.
    order = torch.argsort(d_min)                     # ascending d_min
    n = int(cand.shape[0])
    peak_idx = order[: max(1, int(round(PEAK_PCT * n)))].tolist()
    cap_lo = int(round(CAP_LO_PCT * n))
    cap_hi = max(cap_lo + 1, int(round(CAP_HI_PCT * n)))
    capture_idx = order[cap_lo:cap_hi].tolist()
    far_idx = order[int(round(FAR_PCT * n)):].tolist()

    return {
        "cand": cand, "zstar": zstar, "bw": bw, "d_min": d_min,
        "med_dmin": med_dmin,
        "peak_idx": peak_idx,
        "capture_idx": capture_idx,
        "far_idx": far_idx,
        "world_dim": int(cand.shape[-1]),
    }


def depth_cell(seed: int, depth: float, dose_multiplier: float, geom: Dict, beta: float,
               off_probs: torch.Tensor) -> Dict:
    """Evaluate one (seed, depth) arm. RNG-free (deterministic given geometry).

    depth = effective basin weight (== dose_multiplier * w_ref); dose_multiplier is the
    ordinal dose the dose-response is correlated against.
    """
    cand = geom["cand"]              # [N, H, D]
    zstar = geom["zstar"]           # [D]
    bw = geom["bw"]

    field = ResidueField(ResidueConfig(
        world_dim=WORLD_DIM,
        num_basis_functions=NUM_BASIS,
        kernel_bandwidth=bw,
        accumulation_rate=ACCUMULATION_RATE,
    ))
    field.eval()

    off_baseline_zstar = float(field.evaluate(zstar.reshape(1, -1)).item())  # neural-only

    if depth > 0.0:
        field.accumulate(zstar.reshape(1, -1), harm_magnitude=float(depth))

    with torch.no_grad():
        costs = field.evaluate_trajectory(cand)          # [N]
        probs = softmax_neg(costs, beta)                 # [N]

        # geometry readouts
        peak_val = float(field.evaluate(zstar.reshape(1, -1)).item())
        basin_peak_depth = peak_val - off_baseline_zstar

        # field contrast across the capture band, RELATIVE to peak depth (readiness
        # positive control). Absolute contrast scales with the adaptive weight (tiny),
        # so a relative fraction is the scale-free readout: ~ (1 - exp(-R_CAPTURE_MID^2/2))
        # ~= 0.86 by RBF construction when the basin geometry reaches the capture band.
        gen = torch.Generator().manual_seed(seed * 13 + 5)
        dirs = torch.randn(N_DIRECTIONS, geom["world_dim"], generator=gen)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        probe_pts = zstar.reshape(1, -1) + (R_CAPTURE_MID * bw) * dirs   # [Ndir, D]
        cap_vals = field.evaluate(probe_pts)                              # [Ndir]
        contrast_abs = float((peak_val - cap_vals).abs().mean().item())
        field_contrast_capture = contrast_abs / max(abs(basin_peak_depth), 1e-12) \
            if basin_peak_depth > 0.0 else 0.0

    def band_mass(p: torch.Tensor, idxs: List[int]) -> float:
        if not idxs:
            return 0.0
        return float(p[torch.tensor(idxs)].sum().item())

    # band selection masses under this arm and under OFF (mass conservation: a repeller
    # pushes mass off PEAK and redistributes to everything else, so raw "capture mass
    # change" is confounded by that redistribution -- see capture_vs_far_shift below).
    m_peak = band_mass(probs, geom["peak_idx"])
    m_capture = band_mass(probs, geom["capture_idx"])
    m_far = band_mass(probs, geom["far_idx"])
    off_m_peak = band_mass(off_probs, geom["peak_idx"])
    off_m_capture = band_mass(off_probs, geom["capture_idx"])
    off_m_far = band_mass(off_probs, geom["far_idx"])

    peak_avoidance = off_m_peak - m_peak                  # C1: peak mass lost (unconfounded)
    # C2 (at-a-distance): capture's selection share RELATIVE TO far. Both bands are
    # off-peak, so their relative shift removes the peak-redistribution artifact. Under
    # geometry-mediation capture (closer, higher RBF-tail cost) loses share to far ->
    # shift > 0; under a discrete peak-only gate both are unpenalised -> shift ~ 0.
    denom = m_capture + m_far
    off_denom = off_m_capture + off_m_far
    cap_share = (m_capture / denom) if denom > 0 else 0.0
    off_cap_share = (off_m_capture / off_denom) if off_denom > 0 else 0.0
    capture_vs_far_shift = off_cap_share - cap_share

    row = {
        "seed": seed,
        "depth": float(depth),
        "dose_multiplier": float(dose_multiplier),
        "is_off": bool(dose_multiplier == 0.0),
        "beta": float(beta),
        "bw": float(bw),
        "peak_avoidance": peak_avoidance,
        "capture_vs_far_shift": capture_vs_far_shift,
        "capture_avoidance_net": off_m_capture - m_capture,  # diagnostic (confounded)
        "far_avoidance_net": off_m_far - m_far,              # diagnostic
        "m_peak": m_peak, "m_capture": m_capture, "m_far": m_far,
        "basin_peak_depth": basin_peak_depth,
        "field_contrast_capture": field_contrast_capture,
        "n_peak": len(geom["peak_idx"]),
        "n_capture": len(geom["capture_idx"]),
        "n_far": len(geom["far_idx"]),
        "cost_mean": float(costs.mean().item()),
        "cost_std": float(costs.std(unbiased=False).item()),
        "active_centers": int(field.rbf_field.active_mask.sum().item()),
    }
    return row


def evaluate(seed_rows: Dict[int, List[Dict]]) -> Dict:
    """Aggregate per-seed dose-response, apply readiness then load-bearing criteria."""
    seeds_sorted = sorted(seed_rows.keys())
    doses = list(DOSE_MULTIPLIERS)

    per_seed: List[Dict] = []
    pooled_peak_avoid_at_deepest: List[float] = []
    for seed in seeds_sorted:
        rows = sorted(seed_rows[seed], key=lambda r: r["dose_multiplier"])
        peak_av = [r["peak_avoidance"] for r in rows]
        cap_shift = [r["capture_vs_far_shift"] for r in rows]
        bands_ok = all(
            (r["n_peak"] >= BAND_MIN_COUNT and r["n_capture"] >= BAND_MIN_COUNT
             and r["n_far"] >= BAND_MIN_COUNT)
            for r in rows
        )
        deep_row = rows[-1]  # largest dose_multiplier
        fc_deepest = deep_row["field_contrast_capture"]
        cap_shift_deepest = deep_row["capture_vs_far_shift"]
        peak_deepest = deep_row["peak_avoidance"]

        rho_peak = spearman(doses, peak_av)
        rho_cap = spearman(doses, cap_shift)

        # C1: graded dose-response of PEAK repulsion (monotone in dose AND a real effect).
        c1 = (rho_peak >= SPEARMAN_MONO) and (peak_deepest >= PEAK_EFFECT_FLOOR)
        # C2: geometry-mediated at-a-distance (off-peak CAPTURE candidates lose selection
        # share RELATIVE TO FAR, graded in dose AND a real effect). A discrete peak-only
        # gate leaves capture and far equally unpenalised -> shift ~ 0.
        c2 = (rho_cap >= SPEARMAN_MONO) and (cap_shift_deepest >= CAPTURE_SHIFT_FLOOR)
        ready = fc_deepest >= FIELD_CONTRAST_FLOOR

        pooled_peak_avoid_at_deepest.append(peak_deepest)
        per_seed.append({
            "seed": seed,
            "dose_multipliers": doses,
            "peak_avoidance_by_dose": peak_av,
            "capture_vs_far_shift_by_dose": cap_shift,
            "rho_peak": float(rho_peak),
            "rho_capture_shift": float(rho_cap),
            "peak_avoidance_deepest": float(peak_deepest),
            "capture_vs_far_shift_deepest": float(cap_shift_deepest),
            "field_contrast_capture_deepest": float(fc_deepest),
            "bands_ok": bool(bands_ok),
            "readiness_ok": bool(ready),
            "c1_pass": bool(c1),
            "c2_pass": bool(c2),
            "seed_pass": bool(bands_ok and ready and c1 and c2),
        })

    n_bands_ok = sum(1 for s in per_seed if s["bands_ok"])
    n_ready = sum(1 for s in per_seed if s["readiness_ok"])
    n_c1 = sum(1 for s in per_seed if s["c1_pass"])
    n_c2 = sum(1 for s in per_seed if s["c2_pass"])

    mean_fc = float(sum(s["field_contrast_capture_deepest"] for s in per_seed) / len(per_seed))

    # non-degeneracy net: peak avoidance at the deepest arm must actually spread across
    # seeds (a structurally pinned zero -> degenerate, excluded from scoring).
    degen = check_degeneracy({
        "peak_avoidance_deepest": {"values": pooled_peak_avoid_at_deepest, "floor": PEAK_EFFECT_FLOOR},
    })

    # readiness gate (aggregate positive control -- same statistic C2 depends on)
    preconditions = None
    substrate_not_ready = False
    try:
        preconditions = p0_readiness_gate([
            {"name": "geometry_reaches_capture_band",
             "measured": mean_fc, "threshold": FIELD_CONTRAST_FLOOR, "direction": "lower"},
        ])
    except P0NotReady as exc:
        substrate_not_ready = True
        preconditions = exc.preconditions

    bands_majority = n_bands_ok >= SEED_PASS_N
    c1_pass = n_c1 >= SEED_PASS_N
    c2_pass = n_c2 >= SEED_PASS_N

    if substrate_not_ready or not bands_majority or not degen["non_degenerate"]:
        outcome = "FAIL"
        direction = "non_contributory"
        label = "substrate_not_ready_requeue"
    elif c1_pass and c2_pass:
        outcome = "PASS"
        direction = "supports"
        label = "residue_basin_geometry_graded_and_at_a_distance"
    else:
        outcome = "FAIL"
        direction = "does_not_support"
        label = "no_graded_geometry_mediation_discrete_gate_reading"

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": [
                {"name": "C1_peak_avoidance_monotone_in_depth", "load_bearing": True,
                 "passed": bool(c1_pass)},
                {"name": "C2_capture_avoidance_at_distance_graded", "load_bearing": True,
                 "passed": bool(c2_pass)},
                {"name": "non_vacuity_bands_populated", "load_bearing": False,
                 "passed": bool(bands_majority)},
            ],
            "criteria_non_degenerate": {
                "C1_peak_avoidance_monotone_in_depth": bool(degen["non_degenerate"]),
                "C2_capture_avoidance_at_distance_graded": bool(
                    max(pooled_peak_avoid_at_deepest, default=0.0) > 0.0),
            },
        },
        "aggregates": {
            "seeds": seeds_sorted,
            "dose_multipliers": doses,
            "n_seeds_bands_ok": n_bands_ok,
            "n_seeds_readiness_ok": n_ready,
            "n_seeds_c1_pass": n_c1,
            "n_seeds_c2_pass": n_c2,
            "mean_field_contrast_capture_deepest": mean_fc,
            "per_seed": per_seed,
        },
        "thresholds": {
            "SPEARMAN_MONO": SPEARMAN_MONO,
            "CAPTURE_SHIFT_FLOOR": CAPTURE_SHIFT_FLOOR,
            "PEAK_EFFECT_FLOOR": PEAK_EFFECT_FLOOR,
            "FIELD_CONTRAST_FLOOR": FIELD_CONTRAST_FLOOR,
            "BAND_MIN_COUNT": BAND_MIN_COUNT,
            "SEED_PASS_N": SEED_PASS_N,
            "PEAK_PCT": PEAK_PCT, "CAP_LO_PCT": CAP_LO_PCT, "CAP_HI_PCT": CAP_HI_PCT,
            "FAR_PCT": FAR_PCT, "R_CAPTURE_MID": R_CAPTURE_MID,
        },
        "non_degenerate": degen["non_degenerate"],
        "degeneracy_reason": degen["degeneracy_reason"],
    }


def main(dry_run: bool = False) -> Dict:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})")
    t0 = time.time()
    t0_perf = time.perf_counter()

    seeds = (SEEDS[0], SEEDS[1]) if dry_run else SEEDS
    pool_steps = 90 if dry_run else POOL_STEPS
    n_candidates = 70 if dry_run else N_CANDIDATES
    horizon = 4 if dry_run else HORIZON

    full_config = {
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "world_dim": WORLD_DIM,
        "pool_steps": pool_steps,
        "n_candidates": n_candidates,
        "horizon": horizon,
        "n_directions": N_DIRECTIONS,
        "num_basis_functions": NUM_BASIS,
        "accumulation_rate": ACCUMULATION_RATE,
        "dose_multipliers": list(DOSE_MULTIPLIERS),
        "peak_pct": PEAK_PCT, "cap_lo_pct": CAP_LO_PCT, "cap_hi_pct": CAP_HI_PCT,
        "far_pct": FAR_PCT, "r_capture_mid": R_CAPTURE_MID,
        "seeds": list(seeds),
    }

    all_rows: List[Dict] = []
    seed_rows: Dict[int, List[Dict]] = {}
    total = len(DOSE_MULTIPLIERS)
    for seed in seeds:
        print(f"Seed {seed} Condition mech076_basin_doseresponse")
        reset_all_rng(seed)
        geom = build_seed_geometry(seed, pool_steps, n_candidates, horizon)
        print(f"  [probe] seed={seed} zworld pool+bank built ({pool_steps} steps) "
              f"bw={geom['bw']:.4f} n_peak={len(geom['peak_idx'])} "
              f"n_capture={len(geom['capture_idx'])} n_far={len(geom['far_idx'])}", flush=True)

        # OFF-arm cost spread fixes beta (selection sharpness); a unit-weight basin fixes
        # the per-seed reference weight w_ref so the dose lands in the responsive regime.
        off_field = ResidueField(ResidueConfig(
            world_dim=WORLD_DIM, num_basis_functions=NUM_BASIS,
            kernel_bandwidth=geom["bw"], accumulation_rate=ACCUMULATION_RATE))
        off_field.eval()
        unit_field = ResidueField(ResidueConfig(
            world_dim=WORLD_DIM, num_basis_functions=NUM_BASIS,
            kernel_bandwidth=geom["bw"], accumulation_rate=ACCUMULATION_RATE))
        unit_field.eval()
        unit_field.accumulate(geom["zstar"].reshape(1, -1), harm_magnitude=1.0)
        with torch.no_grad():
            off_costs = off_field.evaluate_trajectory(geom["cand"])     # [N]
            unit_costs = unit_field.evaluate_trajectory(geom["cand"])   # [N] at weight 1.0
        off_std = float(off_costs.std(unbiased=False).item())
        beta = 1.0 / (off_std + 1e-6)
        off_probs = softmax_neg(off_costs, beta)
        # per-peak-candidate cost increment at unit weight -> s_peak; w_ref shifts a peak
        # trajectory's cost by ~1 selection-unit (== off cost std).
        unit_incr = (unit_costs - off_costs)                           # [N]
        if geom["peak_idx"]:
            s_peak = float(unit_incr[torch.tensor(geom["peak_idx"])].mean().item())
        else:
            s_peak = 0.0
        w_ref = (off_std) / max(s_peak, 1e-9)
        if not (w_ref > 0.0):
            w_ref = 1.0
        print(f"  [probe] seed={seed} beta={beta:.3f} s_peak={s_peak:.4f} "
              f"w_ref={w_ref:.4f}", flush=True)

        rows_this_seed: List[Dict] = []
        for di, mult in enumerate(DOSE_MULTIPLIERS):
            print(f"  [probe] seed={seed} depth-arm ep {di + 1}/{total}", flush=True)
            depth = float(mult) * w_ref
            config_slice = dict(full_config)
            config_slice["seed"] = seed
            config_slice["dose_multiplier"] = float(mult)
            config_slice["bw_derived"] = float(geom["bw"])
            config_slice["w_ref_derived"] = float(w_ref)
            with arm_cell(
                seed,
                config_slice=config_slice,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
                # cells within a seed SHARE the per-seed z_world pool, z*, bw, candidate
                # bank, OFF-derived beta and w_ref -> not independent functions of
                # (substrate, config, seed); never reuse across drivers.
                extra_ineligible_reasons=["shares_per_seed_zworld_geometry_across_depth_arms"],
            ) as cell:
                row = depth_cell(seed, depth, float(mult), geom, beta, off_probs)
                row["w_ref"] = float(w_ref)
                cell.stamp(row)
            rows_this_seed.append(row)
            all_rows.append(row)

        seed_rows[seed] = rows_this_seed
        # per-seed verdict
        rows_sorted = sorted(rows_this_seed, key=lambda r: r["dose_multiplier"])
        s_peak_av = [r["peak_avoidance"] for r in rows_sorted]
        s_cap_shift = [r["capture_vs_far_shift"] for r in rows_sorted]
        rho_p = spearman(list(DOSE_MULTIPLIERS), s_peak_av)
        rho_c = spearman(list(DOSE_MULTIPLIERS), s_cap_shift)
        peak_deep = rows_sorted[-1]["peak_avoidance"] if rows_sorted else 0.0
        cap_deep = rows_sorted[-1]["capture_vs_far_shift"] if rows_sorted else 0.0
        fc_deep = rows_sorted[-1]["field_contrast_capture"] if rows_sorted else 0.0
        bands_ok = all(r["n_peak"] >= BAND_MIN_COUNT and r["n_capture"] >= BAND_MIN_COUNT
                       and r["n_far"] >= BAND_MIN_COUNT for r in rows_this_seed)
        seed_pass = bool(bands_ok and fc_deep >= FIELD_CONTRAST_FLOOR
                         and rho_p >= SPEARMAN_MONO and peak_deep >= PEAK_EFFECT_FLOOR
                         and rho_c >= SPEARMAN_MONO and cap_deep >= CAPTURE_SHIFT_FLOOR)
        print(f"  seed={seed} rho_peak={rho_p:.3f} rho_capshift={rho_c:.3f} "
              f"peak_deepest={peak_deep:.4f} capshift_deepest={cap_deep:.4f} "
              f"fc={fc_deep:.3f} bands_ok={bands_ok}")
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'}")

    ev = evaluate(seed_rows)
    outcome = ev["outcome"]
    elapsed = time.time() - t0

    agg = ev["aggregates"]
    print(f"[{EXPERIMENT_TYPE}] label={ev['interpretation']['label']} "
          f"direction={ev['evidence_direction']}")
    print(f"  seeds C1={agg['n_seeds_c1_pass']}/{len(agg['seeds'])} "
          f"C2={agg['n_seeds_c2_pass']}/{len(agg['seeds'])} "
          f"bands_ok={agg['n_seeds_bands_ok']}/{len(agg['seeds'])} "
          f"mean_field_contrast={agg['mean_field_contrast_capture_deepest']:.4f}")
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome} elapsed={elapsed:.1f}s")

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "evidence_class": EVIDENCE_CLASS,
        "started_utc": datetime.utcfromtimestamp(t0).isoformat() + "Z",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "evidence_direction": ev["evidence_direction"],
        "interpretation": ev["interpretation"],
        "aggregates": ev["aggregates"],
        "thresholds": ev["thresholds"],
        "non_degenerate": ev["non_degenerate"],
        "degeneracy_reason": ev["degeneracy_reason"],
        "arm_results": all_rows,
        "wall_independent": True,
        "scope_note": (
            "Tests the basin-geometry leg of MECH-076 only (deep attractor basins as RBF "
            "terrain deformation). Does NOT test over-representation / DG-neurogenesis / "
            "CA3 retraction / palimpsest (unimplemented in V3). evidence_class = "
            "mechanistic_partial."
        ),
        "notes": (
            "Passive terrain readout + terrain-only elite selection (evaluate_trajectory, "
            "ARC-007 strict). Untrained frozen encoder grounds z_world in real substrate "
            "latents (z* and every candidate trajectory are real z_world states). No "
            "planning / action-commitment / training -> wall-independent. Distinct from "
            "INV-053 (binary attractor) and ARC-013/697 (decision separability)."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=dry_run,
        config=full_config,
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=t0_perf,
    )
    print(f"Result written to: {out_path}")
    print(f"Done. Outcome: {outcome}")
    return {"outcome": outcome, "manifest_path": out_path, "run_id": run_id}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run (reduced scope).")
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
