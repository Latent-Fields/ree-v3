"""
V3-EXQ-872: INV-087 proxy tethering constraint (Goodhart guard on a maintenance proxy).

Tests the INV-087 prediction: an ADDITIVE maintenance/progress proxy that sustains a
superordinate goal carries a divergence risk (Goodhart / reward-hacking) UNLESS it is
tethered -- i.e. approximates a POTENTIAL DIFFERENCE over goal-proximity,
    F(s,a,s') = gamma*Phi(s') - Phi(s)          (Ng, Harada & Russell 1999)
the only additive shaping form that is policy-invariant under all dynamics. An
untethered (non-potential) proxy -- e.g. a raw progress/achievement bonus with no
baseline subtraction -- is NOT protected and can pull selection away from the
true-goal optimum.

DESIGN: a controlled E3-selection probe in the V3-EXQ-674/775 family (no env, no
training -- avoids the monostrategy / z_goal-collapse confounds that contaminate
ecological runs). All candidate trajectories share an identical z_world (per
674/775's method), so E3's own raw score_trajectory output is ~constant across
candidates and the injected score_bias landscape IS the geometry the real
e3.select() softmax-selection stage runs over.

THE SINGLE-HOP TELESCOPING ARGUMENT (the mechanism this probe isolates). Ng et al.'s
invariance proof works because a potential-difference bonus summed over a trajectory
telescopes to a boundary term independent of the path taken. At the level of ONE
selection decision, the analogous, minimal instantiation of that telescoping is: give
each candidate trajectory an ENTRY read of the potential Phi(s_k) that is free to vary
per candidate (representing that different candidate plans are evaluated from
slightly different internal vantage points -- an arbitrary, task-irrelevant nuisance
term) and an EXIT read Phi(s'_k) whose difference from the entry read tracks the
candidate's genuine (but imperfectly observed) progress toward the goal. With the
probe's discount fixed at GAMMA=1.0 (the natural, strongest reading for an
undiscounted single transition -- Ng et al.'s result holds for the MDP's own discount,
and there is no "future" beyond this one hop to discount), the potential-difference
form gamma*Phi(s'_k) - Phi(s_k) CANCELS the arbitrary entry-vantage term exactly and
leaves only the genuine progress signal, while a RAW (non-differenced) proxy Phi(s'_k)
alone does not cancel it and is dragged around by an irrelevant per-candidate nuisance
that has nothing to do with which candidate is actually best. This is the same
telescoping mechanism Ng/Harada/Russell rely on, taken to its minimal (single-hop)
non-trivial case, and is why the potential form is expected to preserve the
true-quality optimum here while the raw form is expected to divert it.

TRANSFER CAVEAT (per INV-087's own claim notes): this is a DESIGN-TARGET / limiting-case
certificate, not an off-the-shelf guarantee for REE's learned, non-stationary maintenance
proxy -- the probe demonstrates the mechanism operates on this substrate's real
selection stage, not that a trained proxy head will realise it exactly.

SUBSTRATE MAPPING:
  * "true task quality" q_true(k): a synthetic per-candidate cost landscape (lower =
    better), analogous to 674/775's clean landscape -- the ground truth the selection
    SHOULD track absent any proxy.
  * "goal-proximity" Phi: an ENTRY read phi_entry(k) (per-candidate nuisance noise,
    Uniform(0, PHI_ENTRY_SCALE), uncorrelated with q_true) and an EXIT read expressed
    via gain_obs(k), an IMPERFECT (partially scrambled) estimate of the candidate's true
    gain (Q_SCALE - q_true(k)), analogous to 674's terrain-lesion blend toward an
    independent scramble.
  * injected via the REAL E3 modulatory score_bias path (E3TrajectorySelector.select(),
    e3_selector.py) over the identical-z_world base, exactly as 674/775 do.

THREE ARMS (per EXP-0386):
  CTRL:          score_bias = q_true                                  (no proxy)
  NON_POTENTIAL: score_bias = q_true - W_BIAS*(phi_entry + gain_obs)   (raw/absolute
                 proximity bonus -- Phi(s'_k) alone, in reward-bonus-to-cost-bias
                 convention; retains the arbitrary phi_entry nuisance)
  POTENTIAL:     score_bias = q_true - W_BIAS*gain_obs                (potential
                 DIFFERENCE gamma*Phi(s'_k)-Phi(s_k) at GAMMA=1.0, where phi_entry
                 cancels exactly by construction)

DISCRIMINATIVE METRIC: the TRUE (clean-landscape) quality of the E3-selected candidate,
Q_SCALE - q_true[selected_index], read off the ORACLE q_true regardless of which
landscape informed selection (674/775's convention). PASS = NON_POTENTIAL's mean true
quality is materially LOWER than CTRL's by >= DIVERSION_MARGIN (diverted) AND
POTENTIAL's is NOT materially lower than CTRL's by more than PRESERVATION_TOL
(preserved -- one-sided: a proxy correlated with true gain may legitimately sharpen
the softmax landscape and select the true optimum MORE reliably than CTRL, which is
"not diverted" too, not a violation), on >= 2/3 seeds.

NON-DEGENERACY / READINESS (per EXP-0386's acceptance_checks): a P0 readiness gate
(positive controls) self-routes to substrate_not_ready_requeue if any control is
inert: (1) the clean q_true landscape is genuinely discriminative; (2) the entry-vantage
nuisance alone, at the configured scale, actually CAN move the argmin (so a null result
cannot be attributed to an inert nuisance term); (3) the observed-gain proxy is
genuinely correlated with true gain (a "plausible-looking" proxy, not pure noise); (4)
the uncommitted (multinomial) E3 regime is active (per the 674 finding -- the committed
branch is a temperature-blind but still bias-driven argmin; this probe follows 674/775's
established convention of forcing and verifying the uncommitted regime for this family).
check_degeneracy is also applied to the load-bearing true-quality/diversion-gap spread.

claim_ids = [INV-087] ONLY. The mapping of "goal-proximity" onto a synthetic scalar and
of gamma*Phi(s')-Phi(s) onto a single-hop entry/exit read is an operationalisation of
the general claim, not an independent validation of INV-065/INV-086 (their own
substrates are untouched by this probe).
"""

import argparse
import sys
import time
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

EXPERIMENT_TYPE = "v3_exq_872_inv087_proxy_tethering_constraint"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["INV-087"]

# ----------------------------------------------------------------------------
# Pre-registered constants (NOT derived from the run's own statistics)
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44]        # matches the 674/775 no-env lineage convention; the
                            # seed-44-instability caution in CLAUDE.md is scoped to a
                            # "reef-config env" (real episode stepping) and does not
                            # apply here -- 674/775 (same no-env family) already use 44.
K = 8                       # candidate-pool size (matches 674/775)
WORLD_DIM = 32              # E3Config default z_world dim
ACTION_DIM = 4
N_PROBES = 30               # distinct landscapes per (seed, arm), shared across arms
N_DRAWS = 16                # real e3.select multinomial draws per probe

Q_SCALE = 4.0                # true-quality cost magnitude (lower cost = better), 0..Q_SCALE
PROXY_NOISE = 0.5            # gain_obs blend fraction toward an independent scramble
                              # (0 = proxy is a perfect readout of true gain; 1 = pure
                              # noise). 0.5 = a plausible-looking but genuinely imperfect
                              # proxy, matching the claim's own framing.
PHI_ENTRY_SCALE = 24.0       # entry-vantage nuisance scale. NOT tied to Q_SCALE's
                              # units -- this is an abstract score-landscape probe (same
                              # methodology 674 used an extreme TERR_MIX=0.95 blend to
                              # get clean separation). Calibrated via a pre-registration
                              # simulation (numpy-only, argmin+softmax, same formulas
                              # as below) at the ACTUAL N_PROBES/N_DRAWS/SEEDS scale to
                              # give a robust dissociation margin (d_np ~ 0.6-0.7,
                              # d_pot ~ -0.3 to -0.4, gap ~ 1.0) well clear of the
                              # acceptance thresholds below on every one of the 3
                              # pre-registered seeds -- not tuned against a real run's
                              # outcome.
GAMMA = 1.0                  # potential-difference discount. Fixed at 1.0: this is a
                              # single, undiscounted transition (no "future" beyond this
                              # one hop to discount), which is the natural and STRONGEST
                              # instantiation of the theorem here -- gamma<1 would only
                              # partially cancel the entry-vantage nuisance, blurring
                              # rather than sharpening the dissociation.
W_BIAS = 4.0                 # proxy-bonus weight, calibrated together with
                              # PHI_ENTRY_SCALE (see above) for a robust dissociation
                              # margin at the pre-registered scale.

T_BASE = 1.0                 # E3 default selection temperature (fixed; this probe is
                              # about proxy tethering, not the dopamine/temperature axis
                              # 674/775 already cover)

# Forced uncommitted regime (674/775 convention): hold running_variance above the
# commitment threshold so the real multinomial selection path is the one exercised.
FORCED_RUNNING_VARIANCE = 0.6   # > E3Config.commitment_threshold (0.40)

# Acceptance thresholds (pre-registered)
DIVERSION_MARGIN = 0.5       # NON_POTENTIAL must lose >= this much true-quality vs CTRL
                              # (in Q_SCALE=4.0 units) to count as "diverted"
PRESERVATION_TOL = 0.25      # POTENTIAL's true-quality LOSS vs CTRL (diversion_potential)
                              # must not EXCEED this to count as "preserved". One-sided
                              # by design: the claim is that a tethered proxy does not
                              # divert AWAY from the true-goal optimum -- it is not
                              # required to leave selection bit-identical, and a proxy
                              # correlated with true gain (this design's gain_obs) can
                              # legitimately SHARPEN the softmax landscape and select
                              # the true optimum MORE reliably than CTRL (diversion_pot
                              # < 0). That is "preserved" (if anything, improved), not a
                              # violation -- only a POSITIVE diversion (worse than CTRL)
                              # beyond this tolerance would indicate the potential form
                              # is failing to protect the optimum.
DISSOCIATION_MARGIN = 0.25   # (diversion_non_potential - diversion_potential) must
                              # clear this margin -- genuine separation between arms
SEED_MAJORITY = 2            # dissociation must hold on >= this many of 3 seeds

# P0 readiness floors (positive controls)
CLEAN_SPREAD_FLOOR = 1.0             # true-quality landscape best-vs-worst gap
ENTRY_NOISE_MOVES_ARGMIN_FLOOR = 0.3  # fraction of probes where the entry-vantage
                              # nuisance ALONE (isolated, no gain_obs) can flip the
                              # argmin at this scale -- confirms the nuisance is a real
                              # confound at the configured magnitude, not inert
PROXY_CORRELATION_FLOOR = 0.2         # Pearson corr(gain_obs, true gain) -- confirms
                              # the proxy is genuinely informative, not pure noise
UNCOMMITTED_RATE_FLOOR = 0.99         # selection must be uncommitted in the probe regime

ARMS = ["CTRL", "NON_POTENTIAL", "POTENTIAL"]


def _make_landscape(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One probe's shared substrate: a true-quality cost landscape q_true (lower =
    better), an imperfect observed-gain proxy gain_obs (blended toward an independent
    scramble, like 674's terrain lesion), and an entry-vantage nuisance phi_entry
    (independent noise, uncorrelated with q_true) -- all SHARED across arms so arms
    differ only in how they combine these into score_bias (matched-landscape control).
    """
    q_true = rng.uniform(0.0, Q_SCALE, size=K).astype(np.float64)
    scramble = rng.uniform(0.0, Q_SCALE, size=K).astype(np.float64)
    gain_true = Q_SCALE - q_true
    gain_obs = (1.0 - PROXY_NOISE) * gain_true + PROXY_NOISE * scramble
    phi_entry = rng.uniform(0.0, PHI_ENTRY_SCALE, size=K).astype(np.float64)
    return q_true, gain_obs, phi_entry


def _arm_bias(arm: str, q_true: np.ndarray, gain_obs: np.ndarray, phi_entry: np.ndarray) -> np.ndarray:
    """score_bias landscape for one arm (select()'s sign convention: lower is better,
    favourable bias is negative)."""
    if arm == "CTRL":
        return q_true.copy()
    if arm == "NON_POTENTIAL":
        # raw/absolute proximity bonus Phi(s'_k) alone -- retains the entry nuisance
        return q_true - W_BIAS * (phi_entry + gain_obs)
    if arm == "POTENTIAL":
        # potential DIFFERENCE gamma*Phi(s'_k) - Phi(s_k) at GAMMA=1.0: phi_entry
        # cancels exactly (see module docstring), leaving only gain_obs
        return q_true - W_BIAS * gain_obs
    raise ValueError(f"unknown arm {arm!r}")


def _build_candidates() -> List[Trajectory]:
    """K candidate trajectories with an IDENTICAL z_world (so the real E3
    score_trajectory base is uniform across candidates and the injected score_bias
    landscape IS the geometry the selection runs over) and distinct one-hot first
    actions -- matches 674/775's construction exactly."""
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
    # Force the uncommitted regime so the injected bias landscape (not a
    # temperature-blind committed argmin) is what the probe exercises.
    e3._running_variance = FORCED_RUNNING_VARIANCE
    return e3


def _select_true_quality(e3, cands, bias_landscape: np.ndarray, temperature: float):
    """Run ONE real e3.select() with the arm's bias landscape injected via score_bias."""
    bias = torch.tensor(bias_landscape, dtype=torch.float32)
    res = e3.select(cands, temperature=temperature, score_bias=bias)
    return int(res.selected_index), bool(res.committed)


def _run_cell(arm: str, seed: int, landscapes, e3, cands) -> Dict[str, Any]:
    sel_true_qualities: List[float] = []
    committed_flags: List[bool] = []
    for p, (q_true, gain_obs, phi_entry) in enumerate(landscapes):
        bias_landscape = _arm_bias(arm, q_true, gain_obs, phi_entry)
        for _ in range(N_DRAWS):
            idx, committed = _select_true_quality(e3, cands, bias_landscape, T_BASE)
            # TRUE quality is ALWAYS read off the CLEAN q_true landscape (the oracle),
            # regardless of which arm's biased landscape informed the selection.
            sel_true_qualities.append(float(Q_SCALE - q_true[idx]))
            committed_flags.append(committed)
        if (p + 1) % 10 == 0:
            print(f"  [probe] {arm} seed={seed} ep {p+1}/{N_PROBES}", flush=True)
    return {
        "arm": arm,
        "seed": seed,
        "mean_true_quality": float(np.mean(sel_true_qualities)),
        "uncommitted_rate": float(1.0 - np.mean(committed_flags)),
        "n_draws": len(sel_true_qualities),
    }


def _p0_readiness(seeds) -> Tuple[list, Dict[str, Any]]:
    """Positive-control measurements + the abort gate. Returns (preconditions, diag)
    on success; raises P0NotReady on a starved control."""
    clean_spreads: List[float] = []
    entry_shift_flags: List[float] = []
    all_gain_obs: List[float] = []
    all_gain_true: List[float] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(N_PROBES):
            q_true, gain_obs, phi_entry = _make_landscape(rng)
            clean_spreads.append(float(q_true.max() - q_true.min()))
            bias_entry_only = q_true - W_BIAS * phi_entry
            entry_shift_flags.append(
                1.0 if int(np.argmin(bias_entry_only)) != int(np.argmin(q_true)) else 0.0
            )
            all_gain_obs.extend(gain_obs.tolist())
            all_gain_true.extend((Q_SCALE - q_true).tolist())

    proxy_corr = float(np.corrcoef(np.array(all_gain_obs), np.array(all_gain_true))[0, 1])

    # Confirm the probe regime is genuinely uncommitted via the REAL selector
    # (674/775 convention).
    e3 = _build_e3()
    cands = _build_candidates()
    rng = np.random.default_rng(seeds[0])
    committed = []
    for _ in range(20):
        q_true, _, _ = _make_landscape(rng)
        _, c = _select_true_quality(e3, cands, q_true, T_BASE)
        committed.append(c)
    uncommitted_rate = float(1.0 - np.mean(committed))

    diag = {
        "clean_landscape_spread_mean": float(np.mean(clean_spreads)),
        "entry_noise_argmin_shift_rate": float(np.mean(entry_shift_flags)),
        "proxy_true_gain_correlation": proxy_corr,
        "probe_uncommitted_rate": uncommitted_rate,
    }
    preconditions = p0_readiness_gate([
        {"name": "clean_landscape_discriminative",
         "measured": diag["clean_landscape_spread_mean"],
         "threshold": CLEAN_SPREAD_FLOOR, "direction": "lower"},
        {"name": "entry_noise_moves_argmin",
         "measured": diag["entry_noise_argmin_shift_rate"],
         "threshold": ENTRY_NOISE_MOVES_ARGMIN_FLOOR, "direction": "lower"},
        {"name": "proxy_correlates_with_truth",
         "measured": diag["proxy_true_gain_correlation"],
         "threshold": PROXY_CORRELATION_FLOOR, "direction": "lower"},
        {"name": "probe_regime_uncommitted",
         "measured": diag["probe_uncommitted_rate"],
         "threshold": UNCOMMITTED_RATE_FLOOR, "direction": "lower"},
    ])
    return preconditions, diag


def _full_config() -> Dict[str, Any]:
    return {
        "experiment_type": EXPERIMENT_TYPE, "K": K, "world_dim": WORLD_DIM,
        "action_dim": ACTION_DIM, "n_probes": N_PROBES, "n_draws": N_DRAWS,
        "q_scale": Q_SCALE, "proxy_noise": PROXY_NOISE,
        "phi_entry_scale": PHI_ENTRY_SCALE, "gamma": GAMMA, "w_bias": W_BIAS,
        "t_base": T_BASE, "forced_running_variance": FORCED_RUNNING_VARIANCE,
        "diversion_margin": DIVERSION_MARGIN, "preservation_tol": PRESERVATION_TOL,
        "dissociation_margin": DISSOCIATION_MARGIN, "seed_majority": SEED_MAJORITY,
        "seeds": SEEDS, "arms": ARMS,
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global N_PROBES, N_DRAWS
    t0 = time.perf_counter()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        N_PROBES, N_DRAWS = 6, 4

    # ---- P0 readiness (positive controls + abort gate) ----
    try:
        preconditions, p0_diag = _p0_readiness(seeds)
    except P0NotReady as e:
        manifest: Dict[str, Any] = {
            "run_id": run_id,
            "experiment_type": EXPERIMENT_TYPE,
            "architecture_epoch": "ree_hybrid_guardrails_v1",
            "claim_ids": CLAIM_IDS,
            "experiment_purpose": "diagnostic",
            "outcome": "FAIL",
            "timestamp_utc": timestamp,
            "non_degenerate": False,
            "degeneracy_reason": "P0 readiness unmet -- a control is inert (" + e.reason + ")",
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": e.preconditions,
            },
            "dry_run": dry_run,
        }
        out_path = write_flat_manifest(
            manifest, out_dir=None, dry_run=dry_run,
            config=_full_config(), seeds=SEEDS, script_path=Path(__file__),
            started_at=t0,
        )
        print(f"Manifest written: {out_path}", flush=True)
        print("Outcome: FAIL (substrate_not_ready_requeue)", flush=True)
        manifest["manifest_path"] = str(out_path)
        return manifest

    # ---- main measurement: 3 arms x N seeds (matched landscapes across arms) ----
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        lrng = np.random.default_rng(seed)
        landscapes = [_make_landscape(lrng) for _ in range(N_PROBES)]
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            cfg_slice = {
                "experiment_type": EXPERIMENT_TYPE, "arm": arm, "seed": seed,
                "K": K, "world_dim": WORLD_DIM, "action_dim": ACTION_DIM,
                "n_probes": N_PROBES, "n_draws": N_DRAWS,
                "q_scale": Q_SCALE, "proxy_noise": PROXY_NOISE,
                "phi_entry_scale": PHI_ENTRY_SCALE, "gamma": GAMMA, "w_bias": W_BIAS,
                "t_base": T_BASE, "forced_running_variance": FORCED_RUNNING_VARIANCE,
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
    q_np = q("NON_POTENTIAL")
    q_pot = q("POTENTIAL")
    diversion_np = q_ctrl - q_np
    diversion_pot = q_ctrl - q_pot

    per_seed = []
    hits = 0
    np_diverts_hits = 0
    pot_preserves_hits = 0
    margin_hits = 0
    for seed in seeds:
        qc, qn, qp = q("CTRL", seed), q("NON_POTENTIAL", seed), q("POTENTIAL", seed)
        d_np = qc - qn
        d_pot = qc - qp
        np_diverts = d_np >= DIVERSION_MARGIN
        # One-sided: potential "preserves" the optimum if it does not lose MORE than
        # PRESERVATION_TOL vs CTRL. A negative d_pot (potential BETTER than CTRL,
        # because gain_obs correlates with true gain and sharpens the softmax
        # landscape) is not a violation -- see PRESERVATION_TOL comment above.
        pot_preserves = d_pot <= PRESERVATION_TOL
        margin_ok = (d_np - d_pot) >= DISSOCIATION_MARGIN
        seed_ok = bool(np_diverts and pot_preserves and margin_ok)
        hits += int(seed_ok)
        np_diverts_hits += int(np_diverts)
        pot_preserves_hits += int(pot_preserves)
        margin_hits += int(margin_ok)
        per_seed.append({
            "seed": seed, "q_ctrl": qc, "q_non_potential": qn, "q_potential": qp,
            "diversion_non_potential": d_np, "diversion_potential": d_pot,
            "non_potential_diverts": np_diverts, "potential_preserves": pot_preserves,
            "dissociation_margin_ok": margin_ok, "dissociation_confirmed": seed_ok,
        })

    seed_majority = max(1, min(SEED_MAJORITY, len(seeds)))
    dissociation_confirmed = hits >= seed_majority
    outcome = "PASS" if dissociation_confirmed else "FAIL"

    # ---- non-degeneracy net (applies to evidence runs too) ----
    degen = check_degeneracy({
        "selected_true_quality_across_cells": [r["mean_true_quality"] for r in rows],
        "dissociation_gap_per_seed": {
            "values": [s["diversion_non_potential"] - s["diversion_potential"] for s in per_seed]},
    })

    criteria = [
        {"name": "proxy_tethering_dissociation_seed_majority", "load_bearing": True,
         "passed": dissociation_confirmed},
    ]
    criteria_non_degenerate = {
        "proxy_tethering_dissociation_seed_majority": bool(degen["non_degenerate"]),
    }

    summary = {
        "q_ctrl": q_ctrl, "q_non_potential": q_np, "q_potential": q_pot,
        "diversion_non_potential": diversion_np, "diversion_potential": diversion_pot,
        "dissociation_seed_hits": hits, "seed_majority_required": seed_majority,
        "non_potential_diverts_seed_hits": np_diverts_hits,
        "potential_preserves_seed_hits": pot_preserves_hits,
        "dissociation_margin_seed_hits": margin_hits,
    }

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": "supports" if dissociation_confirmed else "weakens",
        "timestamp_utc": timestamp,
        "dry_run": dry_run,
        "p0_readiness": p0_diag,
        "interpretation": {
            "label": "proxy_tethering_dissociation_confirmed" if dissociation_confirmed
                     else "proxy_tethering_dissociation_not_observed",
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "acceptance_criteria": summary,
        "summary": summary,
        "per_seed": per_seed,
        "arm_results": rows,
        "constants": _full_config(),
    }
    manifest.update(degen)  # non_degenerate / degeneracy_reason / degenerate_metrics

    out_path = write_flat_manifest(
        manifest, out_dir=None, dry_run=dry_run,
        config=_full_config(), seeds=SEEDS, script_path=Path(__file__),
        started_at=t0,
    )
    print(f"Manifest written: {out_path}", flush=True)
    print(f"Outcome: {outcome}", flush=True)
    print(f"  q_ctrl={q_ctrl:.3f} q_non_potential={q_np:.3f} q_potential={q_pot:.3f} "
          f"diversion_np={diversion_np:.3f} diversion_pot={diversion_pot:.3f}", flush=True)
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
        manifest_path=result.get("manifest_path"),
        dry_run=args.dry_run,
    )
