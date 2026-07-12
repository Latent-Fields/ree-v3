"""
V3-EXQ-697: ARC-013 Residue-Separability Falsifier.

Scientific question (Q-079 HA2): Does the residue field (ARC-013 "persistent
latent-space curvature", the E3 Phi_R channel) carry DECISION influence that is
NOT reducible to salience + uncertainty? Is residue a SEPARABLE primitive, or
does it collapse to a re-notation of salience + uncertainty?

This replaces the role the Q-079 / DLIF notes mis-attributed to V3-EXQ-587.
V3-EXQ-587 (EXQ-ISEF-001) tested harm-gradient residue GEOMETRY (formation
speed), ran 2026-05-19, FAIL / non_contributory, and is already reviewed. It
never tested separability. This experiment does.

METHOD (offline re-ranking on a trained agent; rollouts held fixed):
  1. Train an agent (P0 warmup) in the hazard env with harm_gradient on, driving
     agent.update_residue() each step so the ARC-013 residue field populates.
  2. Eval phase (P1): at each decision point log, per candidate trajectory:
       - Phi_R[i] = rho_residue * compute_residue_cost(candidate_i)   (residue cost term)
       - base[i]  = raw_score[i] - Phi_R[i]                           (residue-free primary)
       - salience[i] = z_world visitation-novelty of candidate_i      (residue-INDEPENDENT)
       - uncertainty = E3 _running_variance                           (per-decision scalar)
  3. Post-hoc, pool all (decision, candidate) rows. Fit a reconstruction
     Phi_hat_R = regressor(salience, uncertainty) -- linear AND degree-2 poly,
     report the best recon_R2.
  4. For each decision recompute the E3 argmin three ways on the SAME logged terms:
       INTACT (base + Phi_R), RECON (base + Phi_hat_R), ZEROED (base + 0).
  5. Metrics:
       flip_intact_vs_zeroed  = fraction of decisions where argmin differs (TOTAL residue influence)
       flip_intact_vs_recon   = fraction where argmin differs              (IRREDUCIBLE / separable)
       separability_ratio     = flip_intact_vs_recon / flip_intact_vs_zeroed
       recon_R2               = best reconstruction R^2 of Phi_R from salience + uncertainty

CORRECTNESS NOTE (the key landmine): salience is computed in-script as z_world
visitation-novelty and NEVER touches the residue field, so the reducibility
regression is NOT circular. (The substrate's MECH-314a curiosity defaults to
novelty_source="residue", which WOULD be tautological -- this script does not use
that path.) Vigor (MECH-320) is a per-decision scalar (argmin-invariant within a
decision) and is folded into the per-decision uncertainty term, not the
per-candidate salience -- noted honestly: this measures the field's per-candidate
decision-selection separability, NOT downstream trajectory divergence (a follow-up).

PRE-REGISTERED ACCEPTANCE CRITERIA (single load-bearing gate C1; thresholds fixed here):
  NON-VACUITY gate (else outcome FAIL / evidence_direction non_contributory /
    label substrate_not_ready, NOT a refutation): residue populated
    (mean_weight > 0 AND residue_coverage_pct > 0) AND Phi_R has nonzero
    across-candidate variance AND flip_intact_vs_zeroed > 0 on >= 1 seed.
  C1 PASS (separable; supports ARC-013 as a distinct primitive):
    separability_ratio >= 0.5 in >= 4/5 seeds AND recon_R2 < 0.9.
  C1 FAIL (reducible; does_not_support): separability_ratio <= 0.2 AND recon_R2 >= 0.9.
  Intermediate (0.2 < ratio < 0.5, or mixed): inconclusive -> /failure-autopsy.

INTERPRETATION GRID:
  Outcome                         | Diagnosis / next action
  --------------------------------|-------------------------------------------------------
  C1 PASS                         | ARC-013 residue is a SEPARABLE primitive. The DLIF
                                  |   residue/non-erasure capacity is genuinely the one
                                  |   ingredient NOT covered by existing formalisms
                                  |   (Q-079 corroboration). Build residue as a distinct channel.
  C1 FAIL (reducible)             | Residue reducible to salience+uncertainty. ARC-013
                                  |   residue channel could fold into salience/uncertainty;
                                  |   DLIF reduces fully to existing maths.
  Intermediate / mixed            | Inconclusive -> /failure-autopsy (refine reconstruction
                                  |   family or regime).
  Non-vacuity fail (INVALID)      | Residue has no decision influence in this regime
                                  |   (likely F-dominance per V3-EXQ-571, rho_residue too
                                  |   low, or field empty). Diagnose instrumentation
                                  |   before re-running. NOT a refutation of ARC-013.

claim_ids = [ARC-013]. experiment_purpose = evidence.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

# ------------------------------------------------------------------ #
# Constants                                                          #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_697_arc013_residue_separability_falsifier"
QUEUE_ID = "V3-EXQ-697"
CLAIM_IDS = ["ARC-013"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 44, 45, 46]
P0_WARMUP_EPISODES = 60
EVAL_EPISODES = 25
STEPS_PER_EPISODE = 200

# Residue-independent salience: rolling z_world visitation buffer (MECH-314a
# "visitation" rendering, computed in-script so it never reads the residue field).
VISITATION_BUFFER_MAX = 256

# Pre-registered thresholds.
SEPARABILITY_PASS_RATIO = 0.5
SEPARABILITY_FAIL_RATIO = 0.2
RECON_R2_PASS_CEIL = 0.9
RECON_R2_FAIL_FLOOR = 0.9
PASS_SEEDS_REQUIRED = 4  # of 5

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _obs(obs_dict: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = obs_dict.get(key)
    if v is None:
        return None
    v = v.float()
    if v.dim() == 1:
        v = v.unsqueeze(0)
    return v


def _split_obs(obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _candidate_first_world(agent: REEAgent, cand) -> Optional[np.ndarray]:
    """First-step z_world of a candidate trajectory [world_dim], residue-free read."""
    try:
        wseq = agent.e3._get_world_states(cand)  # [batch, horizon+1, world_dim]
        return wseq[:, 0, :].detach().reshape(-1).cpu().numpy()
    except Exception:
        return None


def _visitation_novelty(first_world: np.ndarray, buffer: np.ndarray) -> float:
    """Min L2 distance from a candidate's first z_world to the visitation buffer."""
    if buffer.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(buffer - first_world[None, :], axis=1)
    return float(d.min())


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.latent.alpha_world = 0.9  # SD-008: z_world fidelity for residue geography
    agent = REEAgent(cfg)
    agent.e3.e3_score_decomp_enabled = True  # need per-candidate residue_weighted term
    return agent


def _fit_recon_r2(
    phi: np.ndarray, salience: np.ndarray, uncertainty: np.ndarray
) -> float:
    """Best R^2 reconstructing Phi_R from [salience, uncertainty]: linear + degree-2."""
    n = phi.shape[0]
    if n < 4:
        return 0.0
    ss_tot = float(((phi - phi.mean()) ** 2).sum())
    if ss_tot <= 1e-12:
        return 0.0  # Phi_R has no variance to reconstruct

    def _r2(feats: np.ndarray) -> float:
        X = np.column_stack([np.ones(n), feats])
        try:
            beta, *_ = np.linalg.lstsq(X, phi, rcond=None)
        except Exception:
            return 0.0
        pred = X @ beta
        ss_res = float(((phi - pred) ** 2).sum())
        return max(0.0, 1.0 - ss_res / ss_tot)

    lin = np.column_stack([salience, uncertainty])
    poly = np.column_stack([
        salience, uncertainty, salience ** 2, uncertainty ** 2, salience * uncertainty
    ])
    return max(_r2(lin), _r2(poly))


def _reconstruct_phi(
    phi: np.ndarray, salience: np.ndarray, uncertainty: np.ndarray
) -> np.ndarray:
    """Best linear reconstruction Phi_hat_R per row (degree-2 features)."""
    n = phi.shape[0]
    feats = np.column_stack([
        salience, uncertainty, salience ** 2, uncertainty ** 2, salience * uncertainty
    ])
    X = np.column_stack([np.ones(n), feats])
    try:
        beta, *_ = np.linalg.lstsq(X, phi, rcond=None)
    except Exception:
        return np.full_like(phi, phi.mean())
    return X @ beta


# ------------------------------------------------------------------ #
# Per-seed run                                                       #
# ------------------------------------------------------------------ #
def _run_seed(seed: int, *, dry_run: bool) -> Dict[str, Any]:
    p0 = 2 if dry_run else P0_WARMUP_EPISODES
    n_eval = 2 if dry_run else EVAL_EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE
    total_eps = p0 + n_eval

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        resource_respawn_on_consume=True,
        harm_gradient_enabled=True,
        harm_gradient_scale=0.30,
    )
    agent = _build_agent(env)
    rho_residue = float(agent.e3.config.rho_residue)

    print(f"Seed {seed} Condition residue_separability", flush=True)

    visitation: List[np.ndarray] = []
    # Pooled per-(decision, candidate) rows collected during eval.
    rows_phi: List[float] = []
    rows_base: List[float] = []
    rows_sal: List[float] = []
    rows_unc: List[float] = []
    rows_decision: List[int] = []
    decision_id = 0

    for ep in range(total_eps):
        is_eval = ep >= p0
        _flat, obs_dict = env.reset()
        agent.reset()
        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        if (ep + 1) % 10 == 0 or ep == total_eps - 1:
            print(
                f"  [train] residue_separability seed={seed} ep {ep + 1}/{total_eps} "
                f"phase={'P1_eval' if is_eval else 'P0'}",
                flush=True,
            )

        for _step in range(steps):
            body, world = _split_obs(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))

            action = agent.select_action(candidates, ticks, temperature=1.0)

            # ---- offline-reranking capture (eval only, multi-candidate ticks) ----
            if is_eval and candidates and len(candidates) >= 2:
                decomp = agent.e3.last_score_decomp or {}
                per_cand = decomp.get("per_candidate", [])
                raw_scores = agent.e3.last_raw_scores
                uncertainty = float(agent.e3._running_variance)
                buf = np.asarray(visitation, dtype=np.float64) if visitation else np.empty((0, wdim))
                if (
                    raw_scores is not None
                    and len(per_cand) == len(candidates)
                    and raw_scores.numel() == len(candidates)
                ):
                    captured_any = False
                    for i, cand in enumerate(candidates):
                        phi_r = float(per_cand[i].get("residue_weighted", 0.0))
                        raw_i = float(raw_scores.reshape(-1)[i].item())
                        fw = _candidate_first_world(agent, cand)
                        if fw is None or not np.isfinite(fw).all():
                            continue
                        sal = _visitation_novelty(fw, buf)
                        rows_phi.append(phi_r)
                        rows_base.append(raw_i - phi_r)
                        rows_sal.append(sal)
                        rows_unc.append(uncertainty)
                        rows_decision.append(decision_id)
                        captured_any = True
                    if captured_any:
                        decision_id += 1

            # advance state
            action_idx = int(action.argmax().item()) % env.action_dim
            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            agent.update_residue(float(harm_signal))

            # update residue-independent visitation buffer with observed z_world
            zw = latent.z_world.detach().reshape(-1).cpu().numpy()
            if np.isfinite(zw).all():
                visitation.append(zw.astype(np.float64))
                if len(visitation) > VISITATION_BUFFER_MAX:
                    visitation.pop(0)

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                _flat, obs_dict = env.reset()
                agent.reset()
                z_self_prev = None
                action_prev = None

    # ---- residue population telemetry ----
    telem = agent.residue_field.get_coverage_telemetry()
    stats = agent.residue_field.get_statistics()
    mean_weight = float(stats["mean_weight"].item())
    coverage = float(telem["residue_coverage_pct"])

    # ---- offline re-ranking analysis ----
    phi = np.asarray(rows_phi, dtype=np.float64)
    base = np.asarray(rows_base, dtype=np.float64)
    sal = np.asarray(rows_sal, dtype=np.float64)
    unc = np.asarray(rows_unc, dtype=np.float64)
    dec = np.asarray(rows_decision, dtype=np.int64)

    n_rows = int(phi.shape[0])
    n_decisions = int(np.unique(dec).shape[0]) if n_rows else 0
    phi_cross_cand_var = 0.0
    recon_r2 = 0.0
    flip_iz = 0.0  # intact vs zeroed
    flip_ir = 0.0  # intact vs recon
    separability_ratio = 0.0

    if n_rows >= 8 and n_decisions >= 2:
        # cross-candidate Phi_R variance (mean within-decision variance)
        within_vars = []
        for d in np.unique(dec):
            m = dec == d
            if m.sum() >= 2:
                within_vars.append(float(np.var(phi[m])))
        phi_cross_cand_var = float(np.mean(within_vars)) if within_vars else 0.0

        recon_r2 = _fit_recon_r2(phi, sal, unc)
        phi_hat = _reconstruct_phi(phi, sal, unc)

        n_flip_iz = 0
        n_flip_ir = 0
        n_dec_used = 0
        for d in np.unique(dec):
            m = dec == d
            if m.sum() < 2:
                continue
            b = base[m]
            p = phi[m]
            ph = phi_hat[m]
            arg_intact = int(np.argmin(b + p))
            arg_zeroed = int(np.argmin(b))
            arg_recon = int(np.argmin(b + ph))
            n_dec_used += 1
            if arg_intact != arg_zeroed:
                n_flip_iz += 1
            if arg_intact != arg_recon:
                n_flip_ir += 1
        if n_dec_used > 0:
            flip_iz = n_flip_iz / n_dec_used
            flip_ir = n_flip_ir / n_dec_used
        separability_ratio = (flip_ir / flip_iz) if flip_iz > 0 else 0.0

    # ---- per-seed verdict ----
    residue_populated = mean_weight > 0.0 and coverage > 0.0
    non_vacuous = (
        residue_populated and phi_cross_cand_var > 0.0 and flip_iz > 0.0 and n_rows >= 8
    )
    c1_pass = (
        non_vacuous
        and separability_ratio >= SEPARABILITY_PASS_RATIO
        and recon_r2 < RECON_R2_PASS_CEIL
    )
    c1_fail_reducible = (
        non_vacuous
        and separability_ratio <= SEPARABILITY_FAIL_RATIO
        and recon_r2 >= RECON_R2_FAIL_FLOOR
    )
    seed_verdict = "PASS" if c1_pass else "FAIL"
    print(f"verdict: {seed_verdict}", flush=True)

    return {
        "seed": seed,
        "mean_weight": mean_weight,
        "residue_coverage_pct": coverage,
        "residue_populated": residue_populated,
        "n_rows": n_rows,
        "n_decisions": n_decisions,
        "phi_cross_candidate_var": phi_cross_cand_var,
        "recon_R2": recon_r2,
        "flip_intact_vs_zeroed": flip_iz,
        "flip_intact_vs_recon": flip_ir,
        "separability_ratio": separability_ratio,
        "non_vacuous": non_vacuous,
        "c1_pass": c1_pass,
        "c1_fail_reducible": c1_fail_reducible,
    }


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    seeds = SEEDS[:2] if dry_run else SEEDS
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run=dry_run))

    n_pass = sum(1 for r in per_seed if r["c1_pass"])
    n_fail_red = sum(1 for r in per_seed if r["c1_fail_reducible"])
    n_nonvacuous = sum(1 for r in per_seed if r["non_vacuous"])
    pass_required = 1 if dry_run else PASS_SEEDS_REQUIRED

    any_nonvacuous = n_nonvacuous >= 1
    overall_pass = any_nonvacuous and n_pass >= pass_required

    if not any_nonvacuous:
        label = "substrate_not_ready_invalid_harness"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            "residue carries no decision influence in this regime "
            "(no seed cleared the non-vacuity gate: populated field + nonzero Phi_R "
            "cross-candidate variance + flip_intact_vs_zeroed > 0)"
        )
    elif overall_pass:
        label = "residue_separable"
        evidence_direction = "supports"
        non_degenerate = True
        degeneracy_reason = ""
    elif n_fail_red >= pass_required:
        label = "residue_reducible_to_salience_uncertainty"
        evidence_direction = "does_not_support"
        non_degenerate = True
        degeneracy_reason = ""
    else:
        label = "inconclusive_route_to_failure_autopsy"
        evidence_direction = "inconclusive"
        non_degenerate = True
        degeneracy_reason = ""

    outcome = "PASS" if overall_pass else "FAIL"

    summary = {
        "n_seeds": len(seeds),
        "n_pass_c1": n_pass,
        "n_fail_reducible": n_fail_red,
        "n_nonvacuous": n_nonvacuous,
        "pass_seeds_required": pass_required,
        "mean_separability_ratio": float(
            np.mean([r["separability_ratio"] for r in per_seed])
        ),
        "mean_recon_R2": float(np.mean([r["recon_R2"] for r in per_seed])),
        "mean_flip_intact_vs_zeroed": float(
            np.mean([r["flip_intact_vs_zeroed"] for r in per_seed])
        ),
        "mean_flip_intact_vs_recon": float(
            np.mean([r["flip_intact_vs_recon"] for r in per_seed])
        ),
    }

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "interpretation": {
            "label": label,
            "method": "offline_reranking_separability",
            "note": (
                "salience is in-script z_world visitation-novelty (residue-INDEPENDENT); "
                "uncertainty is E3 _running_variance; measures per-candidate "
                "decision-selection separability, not downstream trajectory divergence"
            ),
        },
        "acceptance_criteria": {
            "C1_PASS": "separability_ratio >= 0.5 in >= 4/5 seeds AND recon_R2 < 0.9",
            "C1_FAIL_reducible": "separability_ratio <= 0.2 AND recon_R2 >= 0.9",
            "non_vacuity": "populated residue field + nonzero Phi_R cross-candidate variance + flip_intact_vs_zeroed > 0",
        },
        "thresholds": {
            "separability_pass_ratio": SEPARABILITY_PASS_RATIO,
            "separability_fail_ratio": SEPARABILITY_FAIL_RATIO,
            "recon_R2_pass_ceiling": RECON_R2_PASS_CEIL,
            "recon_R2_fail_floor": RECON_R2_FAIL_FLOOR,
            "pass_seeds_required": pass_required,
        },
        "summary": summary,
        "per_seed_results": per_seed,
        "config": {
            "seeds": seeds,
            "p0_warmup_episodes": 2 if dry_run else P0_WARMUP_EPISODES,
            "eval_episodes": 2 if dry_run else EVAL_EPISODES,
            "steps_per_episode": 20 if dry_run else STEPS_PER_EPISODE,
            "harm_gradient_enabled": True,
            "harm_gradient_scale": 0.30,
            "salience_source": "in_script_zworld_visitation_novelty",
        },
        "notes": (
            "ARC-013 residue-separability falsifier. Replaces the role Q-079/DLIF "
            "notes mis-attributed to V3-EXQ-587 (which tested harm-gradient residue "
            "geometry, ran 2026-05-19 FAIL/non_contributory, reviewed). Offline "
            "re-ranking holds rollouts fixed and asks whether residue's per-candidate "
            "decision influence survives reconstruction from salience+uncertainty."
        ),
    }
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("V3-EXQ-697: ARC-013 residue-separability falsifier", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    manifest = run_experiment(args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(
        f"  outcome={manifest['outcome']} direction={manifest['evidence_direction']} "
        f"label={manifest['interpretation']['label']}",
        flush=True,
    )
    print(
        f"  separability_ratio(mean)={manifest['summary']['mean_separability_ratio']:.3f} "
        f"recon_R2(mean)={manifest['summary']['mean_recon_R2']:.3f} "
        f"nonvacuous_seeds={manifest['summary']['n_nonvacuous']}",
        flush=True,
    )
    print(f"  manifest -> {out_path}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
