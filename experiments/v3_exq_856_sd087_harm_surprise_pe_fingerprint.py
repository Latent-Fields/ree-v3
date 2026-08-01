"""
V3-EXQ-856 -- SD-087: harm_surprise_pe OFF vs ON affective fingerprint

Claims: SD-087   (evidence)
EXPERIMENT_PURPOSE = "evidence"

Falsifier for SD-087 ("SD-020's stable reading is scoped to the flag-on
configuration and does not describe default-trained agents"). Re-runs the
V3-EXQ-664 affective fingerprint as a TWO-ARM experiment on the same raw-warmup
substrate, varying ONLY the SD-020 precision-weighted-PE training target for
z_harm_a:

  ARM_OFF : harm_surprise_pe_enabled=False  (the 664 default -- z_harm_a trained
            against the EMA accumulated-harm target SD-020 argues against)
  ARM_ON  : harm_surprise_pe_enabled=True   (z_harm_a trained against the
            precision-weighted affective-surprise target)

DECISIVE READOUTS (the "664 saturation-and-inversion signature"):
  1. within-episode CoV of z_harm_a norm (std/mean over an eval episode's steps),
     averaged over episodes and seeds. SATURATED = CoV below COV_FLOOR.
  2. inverted mode-ordering: mean z_harm_a in shelter > avoid ( > freeze ) --
     "suffering highest when safest". Core inversion = shelter > avoid.

PASS (SD-087 upheld) = the signature (sub-floor CoV AND core inversion) is present
in ARM_OFF and materially reduced in ARM_ON (ARM_ON breaks at least one of the two).
FAIL = the signature survives the flag flip in ARM_ON, relocating the defect to the
encoder or environment (hands the question to Q-086 / V3-EXQ-857).

NON-DEGENERACY (self-routes substrate_not_ready_requeue rather than a verdict):
  * PE-BRANCH DIFFERENTIATION -- the two arms must actually differ on the training
    loss they optimise. _harm_obs_ema is mutated ONLY inside the PE branch
    (agent.py compute_harm_accum_loss), and agent.reset() never clears it, so
    after warmup ARM_ON's _harm_obs_ema must be > PE_EMA_FLOOR and ARM_OFF's must
    be exactly 0.0. If not, ARM_ON silently fell through to the EMA target and the
    manipulation never took -> substrate_not_ready_requeue.
  * OFF REPRODUCES 664 -- ARM_OFF must itself reproduce the saturation+inversion
    signature before ARM_ON means anything. If ARM_OFF does not (e.g. the
    2026-07-31 SD-036 recurrence changed z_harm_a dynamics), there is no baseline
    to reduce -> substrate_not_ready_requeue.

DV-SYMMETRY: the manipulation (harm_surprise_pe_enabled) changes the z_harm_a
TRAINING TARGET and hence the learned affective encoder weights; CoV and per-mode
means are functions of the resulting latent, NOT invariant under any broadcast /
monotone-rescale / permutation of it. So the measured OFF-vs-ON delta is a genuine
measurement, not an arithmetic identity.

SCOPE: raw-warmup only. Every source number (V3-EXQ-664) is from a ~50-episode raw
warmup with scaffold_train_harm_pathway OFF and no scaffolded_sd054_onboarding.
Conclusions are scoped to raw-warmup agents (intake candidate 5 / SD-086/SD-087
scope_note); the OFF arm is kept identical to 664 precisely so it can reproduce
the signature under test.

Output: REE_assembly/evidence/experiments/<run_id>.json (flat manifest).
Estimated runtime: ~75 min on cloud CPU (2 arms x 3 seeds x 50 warmup + 5 eval
x 200 steps, affective stack).
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


EXPERIMENT_TYPE    = "v3_exq_856_sd087_harm_surprise_pe_fingerprint"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS          = ["SD-087"]

SEEDS = [0, 1, 2]

ARMS = [
    {"arm_id": "ARM_OFF", "harm_surprise_pe_enabled": False},  # 664 default (mint)
    {"arm_id": "ARM_ON",  "harm_surprise_pe_enabled": True},
]

# --- pre-registered thresholds (constants; not derived from run statistics) --
COV_FLOOR         = 0.05    # within-episode CoV(z_harm_a) below this = saturated
MIN_MODE_SAMPLES  = 10      # min per-mode step count for an ordering comparison
PE_EMA_FLOOR      = 1e-4    # ARM_ON _harm_obs_ema must exceed this after warmup

# z_goal liveness accumulator (config has z_goal_enabled=True): observed after each
# cell so a dead z_goal stream stays visible in the manifest (V3-EXQ-626/830).
_ZG = ZGoalStreamAccumulator()


# ---------------------------------------------------------------------------
# DV computation from an eval-collection result
# ---------------------------------------------------------------------------

def _within_episode_cov(episodes) -> float:
    """Mean over episodes of std(z_harm_a)/mean(z_harm_a) within the episode."""
    covs = []
    for ep in episodes:
        vals = np.asarray(ep["z_harm_a"], dtype=float)
        if vals.size < 2:
            continue
        m = float(vals.mean())
        if m <= 1e-9:
            covs.append(0.0)   # a pinned-at-zero signal is maximally saturated
            continue
        covs.append(float(vals.std()) / m)
    return float(np.mean(covs)) if covs else 0.0


def _per_mode_means(episodes):
    """Pool all steps across episodes; group z_harm_a by behavioural mode."""
    buckets = {}
    for ep in episodes:
        for v, m in zip(ep["z_harm_a"], ep["mode"]):
            buckets.setdefault(m, []).append(float(v))
    means  = {m: float(np.mean(v)) for m, v in buckets.items()}
    counts = {m: int(len(v)) for m, v in buckets.items()}
    return means, counts


def _signature(cov: float, mode_means, mode_counts):
    """The 664 signature = sub-floor CoV AND core inversion (shelter > avoid).
    Returns (signature_bool, detail_dict)."""
    saturated = cov < COV_FLOOR
    have_shelter = mode_counts.get("shelter", 0) >= MIN_MODE_SAMPLES
    have_avoid   = mode_counts.get("avoid", 0)   >= MIN_MODE_SAMPLES
    have_freeze  = mode_counts.get("freeze", 0)  >= MIN_MODE_SAMPLES

    inversion_core = None
    if have_shelter and have_avoid:
        inversion_core = mode_means["shelter"] > mode_means["avoid"]
    inversion_full = None
    if have_shelter and have_avoid and have_freeze:
        inversion_full = (mode_means["shelter"] > mode_means["avoid"]
                          > mode_means["freeze"])

    sig = bool(saturated and inversion_core is True)
    detail = {
        "saturated_sub_floor_cov": bool(saturated),
        "inversion_core_shelter_gt_avoid": inversion_core,
        "inversion_full_shelter_gt_avoid_gt_freeze": inversion_full,
        "have_shelter": bool(have_shelter),
        "have_avoid": bool(have_avoid),
        "have_freeze": bool(have_freeze),
    }
    return sig, detail


# ---------------------------------------------------------------------------
# Per (arm, seed) cell
# ---------------------------------------------------------------------------

def run_cell(arm, seed, dry_run=False):
    """Runs one (arm, seed) cell and records ONLY raw, threshold-independent
    readouts (CoV, per-mode means/counts, levels, the post-warmup PE EMA). The
    pre-registered thresholds (COV_FLOOR, MIN_MODE_SAMPLES) are applied later in
    run(), OUTSIDE the arm_cell -- so the minted OFF-arm fingerprint records the
    raw substrate readout and a future consumer with different thresholds cannot
    silently HIT a differently-scored cell (arm_reuse_fingerprint_plan.md 7b)."""
    warmup_eps = 3 if dry_run else WARMUP_EPISODES
    eval_eps   = 2 if dry_run else EVAL_EPISODES
    steps      = 30 if dry_run else STEPS_PER_EPISODE

    print(f"\nSeed {seed} Condition {arm['arm_id']}", flush=True)

    agent, env = make_agent_and_env(
        seed,
        env_overrides=None,
        harm_surprise_pe_enabled=arm["harm_surprise_pe_enabled"],
    )
    tag = f" {arm['arm_id']}"
    w = warmup_train(agent, env, warmup_eps, steps, seed, tag=tag)
    ev = eval_collect(agent, env, eval_eps, steps, seed, tag=tag)
    _ZG.observe(agent)   # record z_goal-stream counters (reads them at call time)

    cov = _within_episode_cov(ev["episodes"])
    mode_means, mode_counts = _per_mode_means(ev["episodes"])
    ema = float(w["harm_obs_ema_after_warmup"])

    print(f"[856]{tag} seed={seed} cov(z_harm_a)={cov:.4f} "
          f"harm_obs_ema={ema:.6f} "
          f"mode_means={ {k: round(v,3) for k,v in mode_means.items()} }", flush=True)
    # Per-seed progress verdict is a threshold-free "cell produced a usable
    # readout" proxy; the load-bearing signature is scored in run().
    print(f"verdict: {'PASS' if ev['n_eval_steps'] > 0 else 'FAIL'}", flush=True)

    row = {
        "arm_id": arm["arm_id"],
        "seed": seed,
        "harm_surprise_pe_enabled": bool(arm["harm_surprise_pe_enabled"]),
        "cov_z_harm_a": cov,
        "mean_z_harm_a": float(np.mean([v for ep in ev["episodes"] for v in ep["z_harm_a"]]))
                          if ev["n_eval_steps"] > 0 else 0.0,
        "mean_z_harm_s": float(np.mean([v for ep in ev["episodes"] for v in ep["z_harm_s"]]))
                          if ev["n_eval_steps"] > 0 else 0.0,
        "per_mode_mean_z_harm_a": mode_means,
        "per_mode_counts": mode_counts,
        "harm_obs_ema_after_warmup": ema,
        "eval_mean_reward": ev["mean_reward"],
        "eval_mean_harm": ev["mean_harm"],
        "n_eval_steps": ev["n_eval_steps"],
        "warmup_first10_reward": w["warmup_first10_reward"],
        "warmup_last10_reward": w["warmup_last10_reward"],
    }
    return row


# ---------------------------------------------------------------------------
# Aggregate across seeds within an arm
# ---------------------------------------------------------------------------

def _score_row_signature(row):
    """Apply the pre-registered thresholds to a cell's raw readout (called in
    run(), OUTSIDE the arm_cell, so the fingerprint stays threshold-independent)."""
    sig, detail = _signature(row["cov_z_harm_a"],
                             row["per_mode_mean_z_harm_a"],
                             row["per_mode_counts"])
    row["signature_present"] = bool(sig)
    row["signature_detail"] = detail
    return row


def _arm_summary(arm_id, rows):
    covs = [r["cov_z_harm_a"] for r in rows]
    emas = [r["harm_obs_ema_after_warmup"] for r in rows]
    n_sig = sum(1 for r in rows if r["signature_present"])
    # arm-level signature = majority of seeds show it
    arm_signature = n_sig >= (len(rows) + 1) // 2
    return {
        "arm_id": arm_id,
        "mean_cov_z_harm_a": float(np.mean(covs)) if covs else 0.0,
        "mean_harm_obs_ema": float(np.mean(emas)) if emas else 0.0,
        "n_seeds": len(rows),
        "n_seeds_signature": int(n_sig),
        "arm_signature_present": bool(arm_signature),
    }


def run(seeds=None, dry_run=False):
    if seeds is None:
        seeds = SEEDS
    import time
    t0 = time.perf_counter()

    print(f"[V3-EXQ-856] SD-087 harm_surprise_pe OFF vs ON affective fingerprint\n"
          f"  Seeds: {seeds}  Arms: {[a['arm_id'] for a in ARMS]}\n"
          f"  COV_FLOOR={COV_FLOOR}  MIN_MODE_SAMPLES={MIN_MODE_SAMPLES}  "
          f"PE_EMA_FLOOR={PE_EMA_FLOOR}", flush=True)

    arm_results = []
    for arm in ARMS:
        for seed in seeds:
            # arm_cell: full RNG reset on enter + fingerprint stamp on .stamp().
            # include_driver_script_in_hash=False mints ARM_OFF (the 664-default
            # baseline) reuse-eligible for a future different-driver sibling.
            slice_ = arm_config_slice(
                env_overrides=None,
                harm_surprise_pe_enabled=arm["harm_surprise_pe_enabled"],
            )
            with arm_cell(
                seed,
                config_slice=slice_,
                script_path=Path(__file__),
                config_slice_declared=True,
                include_driver_script_in_hash=False,
            ) as cell:
                row = run_cell(arm, seed, dry_run=dry_run)
                cell.stamp(row)   # fingerprint from config_slice/seed/substrate only
            _score_row_signature(row)   # thresholds applied AFTER stamping
            arm_results.append(row)

    off_rows = [r for r in arm_results if r["arm_id"] == "ARM_OFF"]
    on_rows  = [r for r in arm_results if r["arm_id"] == "ARM_ON"]
    off = _arm_summary("ARM_OFF", off_rows)
    on  = _arm_summary("ARM_ON", on_rows)

    # --- non-degeneracy gates -------------------------------------------------
    off_ema_zero = all(r["harm_obs_ema_after_warmup"] == 0.0 for r in off_rows)
    on_ema_live  = all(r["harm_obs_ema_after_warmup"] > PE_EMA_FLOOR for r in on_rows)
    pe_differentiated = bool(off_ema_zero and on_ema_live)

    off_signature = off["arm_signature_present"]
    on_signature  = on["arm_signature_present"]

    non_degenerate = True
    degeneracy_reason = None
    ng_label = None
    if not pe_differentiated:
        non_degenerate = False
        ng_label = "substrate_not_ready_requeue"
        degeneracy_reason = (
            f"PE branch not differentiated between arms: ARM_OFF _harm_obs_ema all "
            f"zero={off_ema_zero}, ARM_ON _harm_obs_ema all>{PE_EMA_FLOOR}={on_ema_live}. "
            f"ARM_ON must enter the compute_harm_accum_loss PE branch and ARM_OFF must not."
        )
    elif not off_signature:
        non_degenerate = False
        ng_label = "substrate_not_ready_requeue"
        degeneracy_reason = (
            f"ARM_OFF did not reproduce the 664 saturation+inversion signature "
            f"({off['n_seeds_signature']}/{off['n_seeds']} seeds); no baseline to reduce. "
            f"z_harm_a dynamics may have drifted (e.g. 2026-07-31 SD-036 recurrence)."
        )

    # --- verdict (only meaningful when non-degenerate) -----------------------
    if non_degenerate:
        # off_signature is guaranteed True here (NG passed) -> PASS iff ARM_ON
        # materially reduced the signature (does NOT show it).
        passed = bool(off_signature and not on_signature)
        outcome = "PASS" if passed else "FAIL"
        evidence_direction = "supports" if passed else "weakens"
        label = ("sd087_upheld_default_off_pathology" if passed
                 else "sd087_falsified_signature_survives_flag_flip")
    else:
        passed = False
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        label = ng_label

    metrics = {
        "n_seeds": float(len(seeds)),
        "off_mean_cov_z_harm_a": off["mean_cov_z_harm_a"],
        "on_mean_cov_z_harm_a": on["mean_cov_z_harm_a"],
        "off_mean_harm_obs_ema": off["mean_harm_obs_ema"],
        "on_mean_harm_obs_ema": on["mean_harm_obs_ema"],
        "off_n_seeds_signature": float(off["n_seeds_signature"]),
        "on_n_seeds_signature": float(on["n_seeds_signature"]),
        "cov_floor": COV_FLOOR,
        "pe_differentiated": 1.0 if pe_differentiated else 0.0,
        "off_signature_present": 1.0 if off_signature else 0.0,
        "on_signature_present": 1.0 if on_signature else 0.0,
    }

    interpretation = {
        "label": label,
        "arm_off_summary": off,
        "arm_on_summary": on,
        "criteria": [
            {"name": "off_reproduces_664_signature", "load_bearing": True,
             "passed": bool(off_signature)},
            {"name": "on_reduces_signature", "load_bearing": True,
             "passed": bool(not on_signature)},
        ],
        "note": (
            "PASS (SD-087 upheld) = the 664 saturation-and-inversion signature "
            "(within-episode CoV(z_harm_a) < COV_FLOOR AND shelter>avoid mode "
            "ordering) is present in the default-off arm and materially reduced by "
            "the SD-020 PE-target flag. Self-routes substrate_not_ready_requeue if "
            "the PE branch is not differentiated between arms or the OFF arm does "
            "not reproduce the signature. Scoped to raw-warmup agents."
        ),
    }

    result = {
        "experiment_type": EXPERIMENT_TYPE,
        "status": outcome,
        "outcome": outcome,
        "metrics": metrics,
        "arm_results": arm_results,
        "interpretation": interpretation,
        "evidence_direction": evidence_direction,
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
            "cov_floor": COV_FLOOR,
            "min_mode_samples": MIN_MODE_SAMPLES,
            "pe_ema_floor": PE_EMA_FLOOR,
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
