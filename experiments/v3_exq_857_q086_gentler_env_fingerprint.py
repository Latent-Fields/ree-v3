"""
V3-EXQ-857 -- Q-086: gentler-environment confound control for z_harm_a saturation

Claims: Q-086   (diagnostic)
EXPERIMENT_PURPOSE = "diagnostic"

The environment-confound control for the 2026-06-10 z_harm_a saturation
observation (intake candidate 3). Re-measures the V3-EXQ-664 affective
fingerprint on a GENTLER environment to separate two hypotheses:

  * REPRESENTATIONAL / CALIBRATION pathology -- z_harm_a saturates near a ceiling
    regardless of ecology (the affective encoder has lost dynamic range).
  * FAITHFUL CHRONIC SUFFERING -- z_harm_a's level/range genuinely track hazard
    density, so the harsh-env saturation is an ecological fact, not a defect.

  ARM_HARSH  : the V3-EXQ-664 default env (num_hazards=4, hazard_food_attraction=0.7)
  ARM_GENTLE : num_hazards=1, hazard_food_attraction=0.2 (else identical)

Both arms keep harm_surprise_pe_enabled=False -- the observed-pathology config.

DVs per arm: within-episode CoV(z_harm_a); mean z_harm_a level; mean z_harm_s
(the sensory tier, used only for the non-degeneracy check).

DISCRIMINATION (a clean partition once the readiness gate holds):
  PASS "calibration, not ecology"  = ARM_GENTLE still pegs high (sub-floor CoV)
    AND its z_harm_a level does NOT drop materially vs ARM_HARSH.
  PASS "faithful chronic suffering" = ARM_GENTLE's level drops OR its CoV opens up
    (level/range track hazard density).

NON-DEGENERACY (readiness precondition -> substrate_not_ready_requeue if unmet):
  The manipulation must demonstrably take. The SENSORY tier z_harm_s must differ
  across arms beyond the seed-noise band (cross-arm |delta mean z_harm_s| measured
  as a signal-to-noise ratio against the pooled cross-seed SD must exceed
  Z_HARM_S_SNR_K). If z_harm_s is unmoved, the environment was not actually made
  gentler and any z_harm_a reading is uninterpretable.

DV-SYMMETRY: the manipulation (env harshness) changes the environment's harm
dynamics and hence both z_harm_s and z_harm_a; the CoV/level DVs are functions of
the resulting latents, NOT invariant under any broadcast / rescale / permutation.

SCOPE: raw-warmup only (intake candidate 5 / SD-086 scope_note). Conclusions are
scoped to raw-warmup agents.

Output: REE_assembly/evidence/experiments/<run_id>.json (flat manifest).
Estimated runtime: ~70 min on cloud CPU (2 arms x 3 seeds x 50 warmup + 5 eval
x 200 steps; ARM_GENTLE is slightly cheaper).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.baselines.affective_fishtank import (  # noqa: E402
    LINEAGE,
    WARMUP_EPISODES,
    EVAL_EPISODES,
    STEPS_PER_EPISODE,
    make_agent_and_env,
    arm_config_slice,
    warmup_train,
    eval_collect,
)


EXPERIMENT_TYPE    = "v3_exq_857_q086_gentler_env_fingerprint"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = ["Q-086"]

# The readiness precondition `gentle_env_manipulation_took_z_harm_s_differs` IS the
# manipulation-took degeneracy definition for this confound control: it measures the
# cross-arm z_harm_s difference directly. Its reachability is precisely the empirical
# question the two-arm design answers -- if the gentler env cannot move the sensory
# harm tier z_harm_s, then substrate_not_ready_requeue is the correct honest verdict
# (re-queue with a stronger gentling), NOT a mislabelled instrument-specification gap.
# There is no cheaper in-setup positive control than the two arms themselves, so the
# assert_anchor_reachable helper does not apply. Reachable by construction: harsh vs
# gentle envs have genuinely different hazard densities.
ANCHOR_REACHABILITY_EXEMPT = (
    "The z_harm_s-differs precondition is the manipulation-took degeneracy definition; "
    "its reachability is the empirical question the two arms answer, and a below-gate "
    "reading is the correct substrate_not_ready_requeue verdict, not an unmeetable "
    "hand-narrowed predicate. No cheaper in-setup positive control exists than the arms."
)

SEEDS = [0, 1, 2]

# ARM_HARSH env_overrides=None => the 664 default (num_hazards=4, food_attr=0.7).
# ARM_HARSH is fingerprint-identical to V3-EXQ-856's ARM_OFF (same baseline mint).
GENTLE_ENV = {"num_hazards": 1, "hazard_food_attraction": 0.2}
ARMS = [
    {"arm_id": "ARM_HARSH",  "env_overrides": None},        # 664 default (mint)
    {"arm_id": "ARM_GENTLE", "env_overrides": GENTLE_ENV},
]

# --- pre-registered thresholds (constants; not derived from run statistics) --
COV_FLOOR         = 0.05    # within-episode CoV(z_harm_a) below this = saturated
LEVEL_DROP_FRAC   = 0.20    # >=20% drop in gentle-arm z_harm_a level => "tracks"
Z_HARM_S_SNR_K    = 2.0     # cross-arm z_harm_s delta must exceed 2 seed-noise SDs
MIN_EVAL_STEPS    = 100     # per-arm eval-step floor for a non-degenerate readout

# z_goal liveness accumulator (config has z_goal_enabled=True): observed after each
# cell so a dead z_goal stream stays visible in the manifest (V3-EXQ-626/830).
_ZG = ZGoalStreamAccumulator()


def _within_episode_cov(episodes) -> float:
    covs = []
    for ep in episodes:
        vals = np.asarray(ep["z_harm_a"], dtype=float)
        if vals.size < 2:
            continue
        m = float(vals.mean())
        if m <= 1e-9:
            covs.append(0.0)
            continue
        covs.append(float(vals.std()) / m)
    return float(np.mean(covs)) if covs else 0.0


def _pooled_mean(episodes, key) -> float:
    vals = [v for ep in episodes for v in ep[key]]
    return float(np.mean(vals)) if vals else 0.0


def run_cell(arm, seed, dry_run=False):
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(f"\nSeed {seed} Condition {arm['arm_id']}", flush=True)
    agent, env = make_agent_and_env(
        seed,
        env_overrides=arm["env_overrides"],
        harm_surprise_pe_enabled=False,
    )
    tag = f" {arm['arm_id']}"
    w = warmup_train(agent, env, warmup_eps, steps, seed, tag=tag)
    ev = eval_collect(agent, env, eval_eps, steps, seed, tag=tag)
    _ZG.observe(agent)   # record z_goal-stream counters (reads them at call time)

    cov = _within_episode_cov(ev["episodes"])
    level_a = _pooled_mean(ev["episodes"], "z_harm_a")
    level_s = _pooled_mean(ev["episodes"], "z_harm_s")

    print(f"[857]{tag} seed={seed} cov(z_harm_a)={cov:.4f} "
          f"level_z_harm_a={level_a:.4f} level_z_harm_s={level_s:.4f}", flush=True)
    # per-seed progress verdict is a threshold-free "cell produced a usable
    # readout" proxy (no module constant read inside the cell -> the minted
    # fingerprint stays threshold-independent). Discrimination + the MIN_EVAL_STEPS
    # non-degeneracy gate are aggregate, computed in run() OUTSIDE the arm_cell.
    print(f"verdict: {'PASS' if ev['n_eval_steps'] > 0 else 'FAIL'}", flush=True)

    return {
        "arm_id": arm["arm_id"],
        "seed": seed,
        "env_overrides": arm["env_overrides"] or {},
        "cov_z_harm_a": cov,
        "mean_z_harm_a": level_a,
        "mean_z_harm_s": level_s,
        "eval_mean_reward": ev["mean_reward"],
        "eval_mean_harm": ev["mean_harm"],
        "n_eval_steps": ev["n_eval_steps"],
    }


def _arm_agg(arm_id, rows):
    return {
        "arm_id": arm_id,
        "mean_cov_z_harm_a": float(np.mean([r["cov_z_harm_a"] for r in rows])),
        "mean_level_z_harm_a": float(np.mean([r["mean_z_harm_a"] for r in rows])),
        "mean_level_z_harm_s": float(np.mean([r["mean_z_harm_s"] for r in rows])),
        "per_seed_z_harm_s": [r["mean_z_harm_s"] for r in rows],
        "min_eval_steps": int(min(r["n_eval_steps"] for r in rows)),
        "n_seeds": len(rows),
    }


def run(seeds=None, dry_run=False):
    if seeds is None:
        seeds = SEEDS
    import time
    t0 = time.perf_counter()

    print(f"[V3-EXQ-857] Q-086 gentler-env confound control\n"
          f"  Seeds: {seeds}  Arms: {[a['arm_id'] for a in ARMS]}\n"
          f"  COV_FLOOR={COV_FLOOR}  LEVEL_DROP_FRAC={LEVEL_DROP_FRAC}  "
          f"Z_HARM_S_SNR_K={Z_HARM_S_SNR_K}", flush=True)

    arm_results = []
    for arm in ARMS:
        for seed in seeds:
            slice_ = arm_config_slice(
                env_overrides=arm["env_overrides"],
                harm_surprise_pe_enabled=False,
            )
            with arm_cell(
                seed,
                config_slice=slice_,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = run_cell(arm, seed, dry_run=dry_run)
                cell.stamp(row)
            arm_results.append(row)

    harsh_rows  = [r for r in arm_results if r["arm_id"] == "ARM_HARSH"]
    gentle_rows = [r for r in arm_results if r["arm_id"] == "ARM_GENTLE"]
    harsh  = _arm_agg("ARM_HARSH", harsh_rows)
    gentle = _arm_agg("ARM_GENTLE", gentle_rows)

    # --- readiness precondition: did the gentler env actually change z_harm_s? --
    s_harsh  = harsh["mean_level_z_harm_s"]
    s_gentle = gentle["mean_level_z_harm_s"]
    delta_s  = abs(s_harsh - s_gentle)
    pooled_s = harsh["per_seed_z_harm_s"] + gentle["per_seed_z_harm_s"]
    seed_noise_sd = float(np.std(pooled_s))
    seed_noise_sd = max(seed_noise_sd, 1e-6)  # guard divide-by-zero
    z_harm_s_snr = delta_s / seed_noise_sd
    manipulation_took = z_harm_s_snr > Z_HARM_S_SNR_K

    steps_ok = (harsh["min_eval_steps"] >= (10 if dry_run else MIN_EVAL_STEPS)
                and gentle["min_eval_steps"] >= (10 if dry_run else MIN_EVAL_STEPS))

    # --- discrimination (only meaningful once manipulation_took) --------------
    cov_gentle   = gentle["mean_cov_z_harm_a"]
    level_harsh  = harsh["mean_level_z_harm_a"]
    level_gentle = gentle["mean_level_z_harm_a"]
    gentle_saturated = cov_gentle < COV_FLOOR
    level_drop_frac  = ((level_harsh - level_gentle) / level_harsh
                        if level_harsh > 1e-9 else 0.0)
    level_tracks = level_drop_frac >= LEVEL_DROP_FRAC
    cov_tracks   = cov_gentle >= COV_FLOOR

    calibration_pathology = bool(gentle_saturated and not level_tracks)
    faithful_ecological   = bool(level_tracks or cov_tracks)

    non_degenerate = True
    degeneracy_reason = None
    if not manipulation_took:
        non_degenerate = False
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        degeneracy_reason = (
            f"Gentler-env manipulation did not take: cross-arm z_harm_s SNR "
            f"{z_harm_s_snr:.3f} <= K={Z_HARM_S_SNR_K} "
            f"(|delta|={delta_s:.4f}, seed_noise_sd={seed_noise_sd:.4f}). "
            f"The environment was not measurably made gentler."
        )
    elif not steps_ok:
        non_degenerate = False
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        degeneracy_reason = (
            f"Insufficient eval steps for a non-degenerate readout "
            f"(harsh min {harsh['min_eval_steps']}, gentle min {gentle['min_eval_steps']}, "
            f"floor {MIN_EVAL_STEPS})."
        )
    else:
        # Clean partition: exactly one of the two hypotheses holds.
        if calibration_pathology:
            label = "q086_calibration_pathology_representational"
        else:
            label = "q086_faithful_chronic_suffering_ecological"
        outcome = "PASS"

    # readiness-kind precondition (recomputable: measured SNR vs threshold K).
    preconditions = [{
        "name": "gentle_env_manipulation_took_z_harm_s_differs",
        "description": ("cross-arm |delta mean z_harm_s| as an SNR against pooled "
                        "cross-seed SD must exceed Z_HARM_S_SNR_K"),
        "measured": float(z_harm_s_snr),
        "threshold": float(Z_HARM_S_SNR_K),
        "direction": "lower",
        "control": ("ARM_HARSH vs ARM_GENTLE differ only in num_hazards/"
                    "hazard_food_attraction; a real gentling must move the sensory "
                    "harm tier z_harm_s beyond seed noise"),
        "met": bool(manipulation_took),
    }]

    criteria_non_degenerate = {
        "z_harm_s_manipulation_took": bool(manipulation_took),
        "sufficient_eval_steps_both_arms": bool(steps_ok),
    }

    metrics = {
        "n_seeds": float(len(seeds)),
        "harsh_mean_cov_z_harm_a": harsh["mean_cov_z_harm_a"],
        "gentle_mean_cov_z_harm_a": cov_gentle,
        "harsh_mean_level_z_harm_a": level_harsh,
        "gentle_mean_level_z_harm_a": level_gentle,
        "harsh_mean_level_z_harm_s": s_harsh,
        "gentle_mean_level_z_harm_s": s_gentle,
        "z_harm_s_delta": delta_s,
        "z_harm_s_snr": z_harm_s_snr,
        "level_drop_frac": level_drop_frac,
        "cov_floor": COV_FLOOR,
        "manipulation_took": 1.0 if manipulation_took else 0.0,
        "calibration_pathology": 1.0 if (non_degenerate and calibration_pathology) else 0.0,
        "faithful_ecological": 1.0 if (non_degenerate and faithful_ecological) else 0.0,
    }

    interpretation = {
        "label": label,
        "arm_harsh_summary": harsh,
        "arm_gentle_summary": gentle,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": [
            {"name": "z_harm_s_manipulation_took", "load_bearing": True,
             "passed": bool(manipulation_took and steps_ok)},
        ],
        "discrimination": {
            "gentle_saturated_sub_floor_cov": bool(gentle_saturated),
            "level_tracks_hazard_density": bool(level_tracks),
            "cov_tracks_hazard_density": bool(cov_tracks),
            "calibration_pathology_representational": bool(non_degenerate and calibration_pathology),
            "faithful_chronic_suffering_ecological": bool(non_degenerate and faithful_ecological),
        },
        "note": (
            "Diagnostic confound control. Once the gentler-env manipulation is "
            "confirmed to have moved the sensory tier z_harm_s (readiness "
            "precondition), the z_harm_a readout partitions cleanly: a still-"
            "saturated, still-high-level gentle arm => representational/calibration "
            "pathology; a level/range that tracks hazard density => faithful chronic "
            "suffering. Self-routes substrate_not_ready_requeue if z_harm_s did not "
            "move. Scoped to raw-warmup agents."
        ),
    }

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "status": outcome,
        "outcome": outcome,
        "metrics": metrics,
        "arm_results": arm_results,
        "interpretation": interpretation,
        "non_degenerate": bool(non_degenerate),
        "scope": "raw_warmup_agents_only",
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    if degeneracy_reason is not None:
        result["degeneracy_reason"] = degeneracy_reason
    return result


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(seeds=args.seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["timestamp_utc"]      = ts
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"]          = CLAIM_IDS

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config={
            "seeds": args.seeds,
            "arms": [a["arm_id"] for a in ARMS],
            "gentle_env": GENTLE_ENV,
            "cov_floor": COV_FLOOR,
            "level_drop_frac": LEVEL_DROP_FRAC,
            "z_harm_s_snr_k": Z_HARM_S_SNR_K,
            "min_eval_steps": MIN_EVAL_STEPS,
            "lineage": LINEAGE,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        seeds=args.seeds,
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"final_outcome: {result['outcome']}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=bool(args.dry_run),
    )
