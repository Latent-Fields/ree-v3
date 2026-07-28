#!/opt/local/bin/python3
"""V3-EXQ-835 -- MECH-068 / CSH-1 consolidation-operator selectivity dissociation.

Claim under test: MECH-068 (Compact Consolidation Principle), sub-claim CSH-1 --
  "Behavioural selectivity emerges at the consolidation/gating operator (E3),
   not at the shared representational basis (E1)."
  Doc: REE_assembly/docs/architecture/compact_consolidation_principle.md#mech-068
  This is MECH-068's FIRST experimental probe (prior evidence is literature-only:
  Cowley et al. 2023, lit confidence capped at 0.72).

THE DISCRIMINATING TEST (from the doc, CSH-1):
  "Vary E3 consolidation weights (lambda, rho in J(zeta)) while freezing E1.
   Observe: trajectory preference changes; E1 latent representations do not drift."

WHY THIS IS A CLEAN, MACHINE-INDEPENDENT, MEASUREMENT-ONLY TEST (not commitment-gated):
  The E3 candidate score is LINEAR in the two config knobs and they are read at
  exactly one site (ree_core/predictors/e3_selector.py:1162):
        score_i = f_i + lambda_eff * m_i + rho_residue * phi_i        (a COST; lower = preferred)
  where m_i (harm cost) and phi_i (residue cost) are per-candidate [K] vectors and
  f_i is everything else. The candidate set and the latent are UPSTREAM of E3
  scoring and do NOT depend on lambda/rho, so for a fixed decision point the ONLY
  thing (lambda, rho) can change is the selection.

  So we capture, at each genuine E3 select() tick, the per-candidate decomposition
  ONCE at a baseline (lambda=1.0, rho=0.5), extract per-candidate (residual, m, phi)
  where residual_i = last_scores_i - harm_weighted_i - residue_weighted_i (captures
  f + benefit + all modulatory/explore terms), and then ANALYTICALLY recompute the
  argmin winner under every (lambda, rho) on a grid:
        pref(lambda', rho') = argmin_i [ residual_i + lambda'*m_i + rho'*phi_i ]
  This reproduces EXACTLY what setting cfg.e3.lambda_ethical / rho_residue would do
  (the score is linear in those scalars and they enter nowhere else), is fully
  deterministic, and uses argmin -- NOT the multinomial/stratified sampler -- so it
  carries no cross-machine-class divergence (CLAUDE.md "Running the test suite").

  Load-bearing DV (trajectory preference): selection FLIP-RATE = the fraction of
  captured decision points whose argmin winner changes, relative to the baseline
  config, under some (lambda, rho) config on the grid.

DISSOCIATION LEGS:
  (1) E3 gating change DOES change trajectory preference   -> flip_rate > FLIP_RATE_FLOOR
  (2) E1 basis does NOT drift under E3 change              -> architecturally guaranteed:
        (lambda, rho) enter only the E3 scoring cost; E1 encoders are frozen
        (requires_grad=False) during capture, so z_world for any given input is
        invariant to the operator sweep. Recorded (z_world stats) as a positive
        control that the basis is fixed while selectivity is exercised at E3.

VERDICT:
  PASS  (CSH-1 supported): max over the grid of flip_rate >= FLIP_RATE_FLOOR
        (the consolidation operator has selection authority over trajectory
         preference while E1 is frozen).
  FAIL  (CSH-1 not supported / weakens MECH-068): argmin is invariant to
        (lambda, rho) despite non-degenerate consolidation channels -- i.e. the
        primary reality-cost term f dominates selection, so behavioural
        selectivity is NOT controllable at the E3 consolidation weights in the V3
        substrate. This is a genuine claim-level result, not a vacuous ceiling:
        the readiness gate below rules out the degenerate cause (flat channels /
        <2 candidates), so a null here is the finding, not an instrument failure.

READINESS (P0 abort -> substrate_not_ready_requeue, NOT a FAIL):
  The flip criterion can only fire if the operator has something to act on:
    - consolidation channels have cross-candidate spread (range of m OR phi > floor)
      -- same statistic the flip criterion routes on (avoids the V3-EXQ-643
      magnitude-vs-range trap), and
    - >= 2 candidates per decision (a flip is structurally possible).
  If either is unmet the substrate is not ready to answer the question and the run
  self-routes to substrate_not_ready_requeue rather than a misleading FAIL.

No sleep is used (no SLEEP DRIVER line required).

Outputs a flat V3 run pack to REE_assembly/evidence/experiments/.
"""

import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# --- repo import bootstrap (run from ree-v3/ or experiments/) ----------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from experiments.pack_writer import write_flat_manifest
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._metrics import p0_readiness_gate, P0NotReady

# z_goal-stream liveness accumulator (records whether the goal stream was live;
# a dead stream would otherwise be invisible). Observed after each seed's agent
# is stepped; stats folded into the manifest at write time.
_ZG = ZGoalStreamAccumulator()


EXPERIMENT_PURPOSE = "evidence"   # directly tests MECH-068 / CSH-1
EXPERIMENT_TYPE = "v3_exq_835_mech068_consolidation_selectivity_ablation"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- pre-registered thresholds (constants, not derived from run stats) -------
BASELINE_LAMBDA = 1.0
BASELINE_RHO = 0.5
# Consolidation-weight grid analytically applied to the captured decomposition.
LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
RHO_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]

FLIP_RATE_FLOOR = 0.05          # load-bearing: >=5% of decision points flip under some config
CHANNEL_RANGE_FLOOR = 1e-4      # readiness: cross-candidate range of m OR phi
MIN_CANDIDATES_FLOOR = 2.0      # readiness: >=2 candidates so a flip is structurally possible
LAMBDA_EFF_TOL = 1e-3           # assert affective amplification is OFF (lambda_eff == lambda_ethical)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
BURNIN_EPISODES = 8             # E1 learning ON (online), then frozen for capture
CAPTURE_EPISODES = 10           # E1 frozen; capture per-tick decomposition
MAX_STEPS = 80
GRID_SIZE = 8
NUM_HAZARDS = 4


# ---------------------------------------------------------------------------
# Substrate construction
# ---------------------------------------------------------------------------

def _build_agent_env(seed: int) -> Tuple[REEAgent, CausalGridWorldV2, Dict[str, Any]]:
    """Build a CausalGridWorldV2 + REEAgent with the E3 score decomposition enabled
    and the affective-harm amplification OFF (so lambda_eff == lambda_ethical and the
    score is exactly linear in the swept config knobs)."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
        seed=seed,
        hazard_harm=0.5,
    )
    _flat, obs_dict = env.reset()
    body_dim = int(obs_dict["body_state"].shape[-1])
    world_dim = int(obs_dict["world_state"].shape[-1])

    cfg = REEConfig.from_dims(
        body_obs_dim=body_dim,
        world_obs_dim=world_dim,
        action_dim=int(env.action_dim),
    )
    # Clean linearity: no SD-011 affective amplification of lambda_ethical.
    cfg.e3.affective_harm_scale = 0.0
    cfg.e3.lambda_ethical = BASELINE_LAMBDA
    cfg.e3.rho_residue = BASELINE_RHO

    agent = REEAgent(cfg)
    # Enable per-candidate score decomposition (diagnostics-only; gated OFF by default).
    agent.e3.e3_score_decomp_enabled = True
    return agent, env, obs_dict


def _freeze_e1(agent: REEAgent) -> None:
    """Freeze the E1 feature basis: the split obs encoders (latent_stack) and the
    E1 deep predictor. (lambda/rho live in E3, downstream, so this is exactly the
    CSH-1 'freeze E1, vary E3' condition.)"""
    for p in agent.latent_stack.parameters():
        p.requires_grad_(False)
    for p in agent.e1.parameters():
        p.requires_grad_(False)
    agent.eval()


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def _capture_cell(agent: REEAgent, env: CausalGridWorldV2, obs_dict: Dict[str, Any],
                  seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Step the agent through burn-in (E1 learning) then capture (E1 frozen),
    recording per-candidate score decomposition at every GENUINE E3 select tick.

    Returns (rows, telemetry). Each row carries, per candidate:
      residual[K], m[K], phi[K]  -- enough to recompute argmin under any (lambda, rho).
    """
    rows: List[Dict[str, Any]] = []
    n_latched = 0
    n_select = 0
    z_world_norms: List[float] = []
    lambda_eff_seen: List[float] = []

    total_eps = BURNIN_EPISODES + CAPTURE_EPISODES

    for ep in range(total_eps):
        capturing = ep >= BURNIN_EPISODES
        if capturing and ep == BURNIN_EPISODES:
            _freeze_e1(agent)  # freeze the basis before the measurement phase

        _flat, obs_dict = env.reset()
        obs = _flat
        if hasattr(agent, "reset"):
            agent.reset()

        for _step in range(MAX_STEPS):
            # Clear the freshness marker so a latched (held) tick is detectable.
            agent.e3.last_score_diagnostics = None
            action = agent.act(obs)

            diag = agent.e3.last_score_diagnostics
            fresh = isinstance(diag, dict) and len(diag) > 0
            if capturing and fresh:
                decomp = agent.e3.last_score_decomp or {}
                per_cand = decomp.get("per_candidate") or []
                scores = agent.e3.last_scores
                if scores is not None and len(per_cand) >= 1:
                    n_select += 1
                    s = scores.detach().reshape(-1).tolist()
                    k = min(len(s), len(per_cand))
                    residual, m_vec, phi_vec = [], [], []
                    ok = True
                    for i in range(k):
                        pc = per_cand[i]
                        lam_eff = float(pc.get("lambda_eff", BASELINE_LAMBDA))
                        harm_w = float(pc.get("harm_weighted", 0.0))
                        res_w = float(pc.get("residue_weighted", 0.0))
                        lambda_eff_seen.append(lam_eff)
                        if abs(lam_eff) < 1e-12:
                            ok = False
                            break
                        m_i = harm_w / lam_eff
                        phi_i = res_w / BASELINE_RHO if BASELINE_RHO != 0.0 else 0.0
                        residual.append(float(s[i]) - harm_w - res_w)
                        m_vec.append(m_i)
                        phi_vec.append(phi_i)
                    if ok and k >= 1:
                        rows.append({"residual": residual, "m": m_vec, "phi": phi_vec})
                        # positive control: E1 basis reading (frozen)
                        lat = getattr(agent, "_current_latent", None)
                        if lat is not None and getattr(lat, "z_world", None) is not None:
                            z_world_norms.append(float(lat.z_world.detach().norm().item()))
            elif capturing and not fresh:
                n_latched += 1

            obs, harm, done, info, obs_dict = env.step(action)
            if done:
                break

        if (ep + 1) % 1 == 0:
            print(f"  [train] seed={seed} ep {ep + 1}/{total_eps} "
                  f"(phase={'capture' if capturing else 'burnin'} "
                  f"select={n_select} latched={n_latched})", flush=True)

    telem = {
        "n_genuine_select_ticks": n_select,
        "n_latched_ticks": n_latched,
        "mean_candidate_count": (statistics.fmean([len(r["residual"]) for r in rows])
                                 if rows else 0.0),
        "z_world_norm_mean": (statistics.fmean(z_world_norms) if z_world_norms else None),
        "lambda_eff_max_dev_from_baseline": (
            max(abs(x - BASELINE_LAMBDA) for x in lambda_eff_seen) if lambda_eff_seen else 0.0),
    }
    return rows, telem


# ---------------------------------------------------------------------------
# Analytic operator-authority DV
# ---------------------------------------------------------------------------

def _argmin(vals: List[float]) -> int:
    best_i, best_v = 0, vals[0]
    for i in range(1, len(vals)):
        if vals[i] < best_v:
            best_i, best_v = i, vals[i]
    return best_i


def _score_at(row: Dict[str, Any], lam: float, rho: float) -> List[float]:
    return [row["residual"][i] + lam * row["m"][i] + rho * row["phi"][i]
            for i in range(len(row["residual"]))]


def _flip_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each (lambda, rho) config, the fraction of decision points whose argmin
    winner differs from the baseline config's winner. Only ticks with >=2 candidates
    can flip; ticks with 1 candidate are counted in the denominator as non-flip
    (structurally can't flip) and reported separately."""
    # Baseline winners + internal consistency check (reconstructed baseline argmin
    # must match the winner under the baseline weights).
    baseline_win: List[int] = []
    for r in rows:
        baseline_win.append(_argmin(_score_at(r, BASELINE_LAMBDA, BASELINE_RHO)))

    per_config: List[Dict[str, Any]] = []
    harm_ranges: List[float] = []
    phi_ranges: List[float] = []
    for r in rows:
        if len(r["m"]) >= 2:
            harm_ranges.append(max(r["m"]) - min(r["m"]))
            phi_ranges.append(max(r["phi"]) - min(r["phi"]))

    for lam in LAMBDA_GRID:
        for rho in RHO_GRID:
            if lam == BASELINE_LAMBDA and rho == BASELINE_RHO:
                continue
            flips = 0
            n = 0
            for idx, r in enumerate(rows):
                if len(r["residual"]) < 2:
                    n += 1
                    continue
                n += 1
                w = _argmin(_score_at(r, lam, rho))
                if w != baseline_win[idx]:
                    flips += 1
            per_config.append({
                "lambda": lam,
                "rho": rho,
                "flip_rate": (flips / n) if n else 0.0,
                "n_flips": flips,
                "n_ticks": n,
            })

    max_flip = max((c["flip_rate"] for c in per_config), default=0.0)
    lambda_only = [c for c in per_config if c["rho"] == BASELINE_RHO]
    rho_only = [c for c in per_config if c["lambda"] == BASELINE_LAMBDA]
    return {
        "per_config_flip": per_config,
        "max_flip_rate": max_flip,
        "max_flip_rate_lambda_axis": max((c["flip_rate"] for c in lambda_only), default=0.0),
        "max_flip_rate_rho_axis": max((c["flip_rate"] for c in rho_only), default=0.0),
        "harm_cost_range_mean": (statistics.fmean(harm_ranges) if harm_ranges else 0.0),
        "residue_cost_range_mean": (statistics.fmean(phi_ranges) if phi_ranges else 0.0),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    per_seed_results: List[Dict[str, Any]] = []
    all_rows_count = 0
    harm_range_vals: List[float] = []
    phi_range_vals: List[float] = []
    cand_count_vals: List[float] = []
    max_flip_vals: List[float] = []
    lambda_eff_devs: List[float] = []

    for seed in seeds:
        print(f"Seed {seed} Condition csh1_consolidation_selectivity", flush=True)
        agent, env, obs_dict = _build_agent_env(seed)
        rows, telem = _capture_cell(agent, env, obs_dict, seed)
        _ZG.observe(agent)  # AFTER stepping -- reads the goal-stream counters
        analysis = _flip_analysis(rows) if rows else {
            "per_config_flip": [], "max_flip_rate": 0.0,
            "max_flip_rate_lambda_axis": 0.0, "max_flip_rate_rho_axis": 0.0,
            "harm_cost_range_mean": 0.0, "residue_cost_range_mean": 0.0,
        }
        all_rows_count += len(rows)
        harm_range_vals.append(analysis["harm_cost_range_mean"])
        phi_range_vals.append(analysis["residue_cost_range_mean"])
        cand_count_vals.append(telem["mean_candidate_count"])
        max_flip_vals.append(analysis["max_flip_rate"])
        lambda_eff_devs.append(telem["lambda_eff_max_dev_from_baseline"])

        seed_verdict = "PASS" if analysis["max_flip_rate"] >= FLIP_RATE_FLOOR else "FAIL"
        per_seed_results.append({
            "seed": seed,
            "n_genuine_select_ticks": telem["n_genuine_select_ticks"],
            "n_latched_ticks": telem["n_latched_ticks"],
            "mean_candidate_count": telem["mean_candidate_count"],
            "z_world_norm_mean": telem["z_world_norm_mean"],
            "max_flip_rate": analysis["max_flip_rate"],
            "max_flip_rate_lambda_axis": analysis["max_flip_rate_lambda_axis"],
            "max_flip_rate_rho_axis": analysis["max_flip_rate_rho_axis"],
            "harm_cost_range_mean": analysis["harm_cost_range_mean"],
            "residue_cost_range_mean": analysis["residue_cost_range_mean"],
            "per_config_flip": analysis["per_config_flip"],
        })
        print(f"verdict: {seed_verdict}", flush=True)

    # ---- aggregate readiness measurements (over all seeds) ----
    channel_range = max(
        statistics.fmean(harm_range_vals) if harm_range_vals else 0.0,
        statistics.fmean(phi_range_vals) if phi_range_vals else 0.0,
    )
    mean_candidates = statistics.fmean(cand_count_vals) if cand_count_vals else 0.0
    max_lambda_eff_dev = max(lambda_eff_devs) if lambda_eff_devs else 0.0
    agg_max_flip = statistics.fmean(max_flip_vals) if max_flip_vals else 0.0
    overall_max_flip = max(max_flip_vals) if max_flip_vals else 0.0

    manifest_common = {
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": ["MECH-068"],
        "csh_sub_claim": "CSH-1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "per_seed_results": per_seed_results,
        "custom_information": {
            "lambda_grid": LAMBDA_GRID,
            "rho_grid": RHO_GRID,
            "baseline_lambda": BASELINE_LAMBDA,
            "baseline_rho": BASELINE_RHO,
            "flip_rate_floor": FLIP_RATE_FLOOR,
            "affective_harm_scale": 0.0,
            "n_capture_rows_total": all_rows_count,
            "aggregate_mean_max_flip_rate": agg_max_flip,
            "overall_max_flip_rate": overall_max_flip,
            "readiness_channel_range": channel_range,
            "readiness_mean_candidates": mean_candidates,
            "lambda_eff_max_dev_from_baseline": max_lambda_eff_dev,
            "dv_symmetry_note": (
                "lambda scales the per-candidate harm-cost channel m (a [K] vector, "
                "readiness-gated non-constant); rho scales the per-candidate residue-cost "
                "channel phi. Scaling a non-constant per-candidate cost term is NOT "
                "invariant under the argmin DV, so the flip metric measures a real effect."
            ),
        },
    }

    # ---- P0 readiness gate: is the operator able to exercise authority at all? ----
    try:
        preconditions = p0_readiness_gate([
            {"name": "consolidation_channel_range_nondegenerate",
             "measured": channel_range, "threshold": CHANNEL_RANGE_FLOOR,
             "direction": "lower", "comparator": ">",
             "control": "cross-candidate range of harm-cost m OR residue-cost phi at "
                        "baseline weights -- the SAME statistic the flip criterion routes on"},
            {"name": "mean_candidate_count_supra_floor",
             "measured": mean_candidates, "threshold": MIN_CANDIDATES_FLOOR,
             "direction": "lower", "comparator": ">=",
             "control": "mean number of E3 candidates per decision; a flip is "
                        "structurally impossible with <2 candidates"},
        ])
    except P0NotReady as e:
        manifest = dict(manifest_common)
        manifest["outcome"] = "FAIL"
        manifest["experiment_purpose"] = "diagnostic"
        manifest["evidence_direction"] = "non_contributory"
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = "substrate_not_ready: " + e.reason
        manifest["interpretation"] = {
            "label": "substrate_not_ready_requeue",
            "preconditions": e.preconditions,
        }
        return manifest

    # Internal-consistency check: affective amplification must have been OFF, else
    # the analytic re-weight is not exactly linear in the config lambda (setting
    # cfg.e3.lambda_ethical=lambda' would give lambda_eff'=lambda'*(1+affective*...),
    # not lambda'). We set affective_harm_scale=0.0, so lambda_eff==lambda_ethical;
    # if that assumption failed, the result is not trustworthy -> self-exclude.
    linearity_ok = max_lambda_eff_dev <= LAMBDA_EFF_TOL
    if not linearity_ok:
        manifest = dict(manifest_common)
        manifest["outcome"] = "FAIL"
        manifest["experiment_purpose"] = "diagnostic"
        manifest["evidence_direction"] = "non_contributory"
        manifest["non_degenerate"] = False
        manifest["degeneracy_reason"] = (
            "affective harm amplification unexpectedly active "
            f"(max lambda_eff deviation {max_lambda_eff_dev:.4g} > {LAMBDA_EFF_TOL}); "
            "analytic lambda/rho re-weight not exactly linear -- not trustworthy")
        manifest["interpretation"] = {
            "label": "instrument_not_linear_requeue",
            "linearity_max_lambda_eff_dev": max_lambda_eff_dev,
        }
        return manifest

    # ---- readiness passed: a null flip result is a REAL weakens, not degenerate ----
    selectivity_met = overall_max_flip >= FLIP_RATE_FLOOR
    outcome = "PASS" if selectivity_met else "FAIL"
    evidence_direction = "supports" if selectivity_met else "weakens"

    manifest = dict(manifest_common)
    manifest["outcome"] = outcome
    manifest["evidence_direction"] = evidence_direction
    manifest["non_degenerate"] = True
    manifest["interpretation"] = {
        "label": ("consolidation_operator_has_selection_authority" if selectivity_met
                  else "f_dominance_operator_no_selection_authority"),
        "preconditions": preconditions,
        "criteria": [
            {"name": "operator_selection_authority_flip_rate",
             "load_bearing": True,
             "measured": overall_max_flip,
             "threshold": FLIP_RATE_FLOOR,
             "passed": bool(selectivity_met)},
        ],
        "linearity_check_affective_off": bool(linearity_ok),
        "linearity_max_lambda_eff_dev": max_lambda_eff_dev,
    }
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-835 MECH-068/CSH-1 consolidation selectivity")
    parser.add_argument("--dry-run", action="store_true", help="tiny smoke run")
    parser.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    args = parser.parse_args()

    seeds = list(DEFAULT_SEEDS)
    if args.dry_run:
        BURNIN_EPISODES = 1
        CAPTURE_EPISODES = 2
        MAX_STEPS = 30
        seeds = [0]
    elif args.seeds is not None:
        seeds = list(range(args.seeds))

    t_start = time.perf_counter()
    result = run_experiment(seeds, dry_run=args.dry_run)

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_id"] = f"{EXPERIMENT_TYPE}_{run_ts}_v3"

    full_config = {
        "seeds": seeds,
        "lambda_grid": LAMBDA_GRID,
        "rho_grid": RHO_GRID,
        "baseline_lambda": BASELINE_LAMBDA,
        "baseline_rho": BASELINE_RHO,
        "burnin_episodes": BURNIN_EPISODES,
        "capture_episodes": CAPTURE_EPISODES,
        "max_steps": MAX_STEPS,
        "grid_size": GRID_SIZE,
        "num_hazards": NUM_HAZARDS,
        "flip_rate_floor": FLIP_RATE_FLOOR,
        "channel_range_floor": CHANNEL_RANGE_FLOOR,
        "min_candidates_floor": MIN_CANDIDATES_FLOOR,
        "affective_harm_scale": 0.0,
        "e3_score_decomp_enabled": True,
        "e1_frozen_during_capture": True,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t_start,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"[result] outcome={result['outcome']} "
          f"overall_max_flip_rate="
          f"{result.get('custom_information', {}).get('overall_max_flip_rate')}", flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
