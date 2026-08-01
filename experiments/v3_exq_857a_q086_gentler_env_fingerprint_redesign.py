"""
V3-EXQ-857a -- Q-086: gentler-environment confound control REDESIGN (successor to V3-EXQ-857)

Claims: Q-086   (diagnostic)
EXPERIMENT_PURPOSE = "diagnostic"

V3-EXQ-857 (run v3_exq_857_q086_gentler_env_fingerprint_20260801T134757Z_v3) tested
the same question with 2 arms (num_hazards 4 vs 1) and 3 seeds, and its own
readiness precondition (gentle_env_manipulation_took_z_harm_s_differs) FAILED:
cross-arm z_harm_s SNR 0.42 against a required 2.0 (|delta mean z_harm_s|=0.072 vs
seed_noise_sd=0.171). Confirmed by
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-857_2026-08-01.md (status:
confirmed), diagnosed as a POWER problem (3 seeds, high per-seed variance), not a
wrong hypothesis, and routed explicitly to:

  /queue-experiment a redesigned re-run with either (a) more seeds (5-10) ... or
  (b) a starker environmental gradient (e.g. num_hazards 4 vs 0), or both, plus
  consider adding episode-survival-time as an explicit secondary DV.

This script implements (a)+(b)+the survival-time DV together (the autopsy's "or
both" framing), per the routing recorded on claims.yaml Q-086 (2026-08-01T17:22Z
governance cycle, REE_assembly 5c7e5e1134).

ARM_HARSH  : the V3-EXQ-664/856/857 default env (num_hazards=4, hazard_food_attraction=0.7)
ARM_GENTLE : num_hazards=1, hazard_food_attraction=0.2 (857's original gentle arm,
             kept for continuity/comparability across the two attempts)
ARM_BENIGN : num_hazards=0, hazard_food_attraction=0.0 (NEW -- the starker gradient
             the autopsy routed to; a true hazard-free floor. Verified against
             current ree_core/environment/causal_grid_world.py source (2026-08-01):
             num_hazards=0 means self.hazards stays empty for the whole episode
             (placement is `for _ in range(min(self.num_hazards, len(forage_pool)))`
             at init/reset; every subsequent hazard-drift/move/food-attraction
             branch is gated on `if self.hazards:` and background_drift_enabled
             defaults False and is not set by this baseline's HARSH_ENV_KWARGS --
             so no hazard is ever added independent of num_hazards). No edge case;
             this is a genuine hazard-free floor, not an untested config.

Both arms keep harm_surprise_pe_enabled=False -- the observed-pathology config
(unchanged from 857).

DVs per arm: within-episode CoV(z_harm_a); mean z_harm_a level; mean z_harm_s (the
sensory tier, used for the non-degeneracy check); NEW -- mean_episode_survival_steps
(mean/min per-episode step count at termination), directly instrumenting the signal
857's autopsy flagged as uncounted: min_eval_steps was 58 (harsh) vs 125 (gentle) in
857, suggesting harsh-arm episodes terminate earlier (plausibly hazard-collision-
driven) in a way the per-step z_harm_s average alone doesn't capture. Episode length
is already available from eval_collect's per-episode step lists (len(ep["z_harm_a"])
== steps survived before `done` or the STEPS_PER_EPISODE cap) -- no change to the
shared baseline lib is needed.

DISCRIMINATION (a clean partition once the readiness gate holds):
  PASS "calibration, not ecology"  = ARM_BENIGN still pegs high (sub-floor CoV) AND
    its z_harm_a level does NOT drop materially vs ARM_HARSH.
  PASS "faithful chronic suffering" = ARM_BENIGN's level drops OR its CoV opens up
    (level/range track hazard density).
  ARM_GENTLE is reported throughout for the 3-point (4/1/0) gradient and cross-
  attempt comparability, but does NOT gate the self-route (see below).

NON-DEGENERACY (readiness precondition -> substrate_not_ready_requeue if unmet):
  Precondition 1 (857's original statistic, re-targeted to the STARKEST pair): the
  SENSORY tier z_harm_s must differ across ARM_HARSH vs ARM_BENIGN beyond the
  seed-noise band (SNR against the pooled cross-seed SD of those two arms must
  exceed Z_HARM_S_SNR_K). ARM_HARSH vs ARM_GENTLE's SNR (857's original pair) is
  also computed and reported for comparability but is NOT load-bearing here -- the
  starkest pair is the correct place to key the self-route, since a starker
  gradient is exactly what a power-limited manipulation needs to clear the gate.
  Precondition 2 (NEW, non-gating secondary check): mean_episode_survival_steps
  must differ across ARM_HARSH vs ARM_BENIGN beyond seed noise (same SNR-style
  test). Reported explicitly rather than folded into the primary verdict -- if
  survival time moves even where z_harm_s does not, that is itself informative.
  If precondition 1 fails even at num_hazards=0 vs 4 and n=6 seeds: this is a
  strong signal the sensory-tier z_harm_s readout itself may not be ecologically
  sensitive at this warmup scale (distinct from "just underpowered") -- flagged
  explicitly in degeneracy_reason rather than silently self-routing a second time.

DV-SYMMETRY: the manipulation (env harshness / hazard count) changes the
environment's harm dynamics and hence both z_harm_s, z_harm_a, and episode
survival time; none of the CoV/level/survival DVs are invariant under any
broadcast / rescale / permutation of the manipulation.

SCOPE: raw-warmup only (intake candidate 5 / SD-086 scope_note), same as 857.
Conclusions are scoped to raw-warmup agents.

GOV-REUSE-1 (checked 2026-08-01): decisive readout = cross-arm z_harm_s SNR at
num_hazards 4-vs-0. `REE_assembly/scripts/reanalysis_query.py query --readout
z_harm_s --claim Q-086` returns exactly 1 manifest (V3-EXQ-857 itself, substrate_hash
690c1ccd29756988...), which carries only ARM_HARSH (num_hazards=4) and ARM_GENTLE
(num_hazards=1) cells -- no num_hazards=0 arm exists anywhere in the corpus. Not
recoverable by reanalysis -> a new ARM_BENIGN cell is required, proceed to author.

Re-derive brake (Step 2.5b, checked 2026-08-01): 1 counted substrate_ceiling-family
autopsy on Q-086 (failure_autopsy_V3-EXQ-857_2026-08-01, category
non_contributory/precondition_unmet -- an INSTRUMENT/readiness defect that owes no
substrate build, not a substrate_ceiling verdict), well under threshold=2. Brake
does not fire.

ARM_HARSH is fingerprint-identical to V3-EXQ-856's ARM_OFF, V3-EXQ-857's ARM_HARSH,
and V3-EXQ-664's default (reuse-eligible per the lineage's mint-as-you-go
convention; include_driver_script_in_hash=False already set on the affective_fishtank
lineage module).

Output: REE_assembly/evidence/experiments/<run_id>.json (flat manifest).
Estimated runtime: ~3 arms x 6 seeds x (50 warmup + 5 eval x 200 steps) = 18
arm-seed cells vs 857's 6 (2 arms x 3 seeds) -- roughly 3x the cells; budgeted
~230 min on cloud CPU (857's own elapsed_seconds was ~70 min for 6 cells).
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


EXPERIMENT_TYPE    = "v3_exq_857a_q086_gentler_env_fingerprint_redesign"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS          = ["Q-086"]

# The readiness precondition `gentle_env_manipulation_took_z_harm_s_differs_starkest`
# IS the manipulation-took degeneracy definition for this confound control, keyed to
# the starkest pair (ARM_HARSH vs ARM_BENIGN): its reachability is precisely the
# empirical question the redesign answers -- if even num_hazards=0 vs 4 cannot move
# the sensory harm tier z_harm_s beyond seed noise at n=6 seeds, that is itself an
# informative finding about the readout, not an unmeetable hand-narrowed predicate.
# There is no cheaper in-setup positive control than the arms themselves.
ANCHOR_REACHABILITY_EXEMPT = (
    "The z_harm_s-differs precondition (starkest pair) is the manipulation-took "
    "degeneracy definition; its reachability is the empirical question this redesign "
    "answers, and a below-gate reading at the starkest gradient is itself informative "
    "(flagged in degeneracy_reason), not an unmeetable hand-narrowed predicate. No "
    "cheaper in-setup positive control exists than the arms."
)

SEEDS = [0, 1, 2, 3, 4, 5]

# ARM_HARSH env_overrides=None => the 664 default (num_hazards=4, food_attr=0.7).
GENTLE_ENV = {"num_hazards": 1, "hazard_food_attraction": 0.2}
BENIGN_ENV = {"num_hazards": 0, "hazard_food_attraction": 0.0}
ARMS = [
    {"arm_id": "ARM_HARSH",  "env_overrides": None},        # 664 default (mint)
    {"arm_id": "ARM_GENTLE", "env_overrides": GENTLE_ENV},  # 857's original gentle arm
    {"arm_id": "ARM_BENIGN", "env_overrides": BENIGN_ENV},  # NEW starker floor
]

# --- pre-registered thresholds (constants; not derived from run statistics) --
COV_FLOOR         = 0.05    # within-episode CoV(z_harm_a) below this = saturated
LEVEL_DROP_FRAC   = 0.20    # >=20% drop in benign-arm z_harm_a level => "tracks"
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


def _episode_survival_steps(episodes) -> list:
    """Per-episode step count at termination (done OR the STEPS_PER_EPISODE cap).
    eval_collect breaks its inner loop on `done`, so len(ep["z_harm_a"]) is exactly
    the number of steps that episode ran -- no change to the shared baseline lib
    needed to derive this NEW secondary DV."""
    return [len(ep["z_harm_a"]) for ep in episodes]


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
    survival_steps = _episode_survival_steps(ev["episodes"])
    mean_survival = float(np.mean(survival_steps)) if survival_steps else 0.0
    min_survival = int(min(survival_steps)) if survival_steps else 0

    print(f"[857a]{tag} seed={seed} cov(z_harm_a)={cov:.4f} "
          f"level_z_harm_a={level_a:.4f} level_z_harm_s={level_s:.4f} "
          f"mean_survival_steps={mean_survival:.1f}", flush=True)
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
        "episode_survival_steps": survival_steps,
        "mean_episode_survival_steps": mean_survival,
        "min_episode_survival_steps": min_survival,
    }


def _arm_agg(arm_id, rows):
    all_survival = [s for r in rows for s in r["episode_survival_steps"]]
    return {
        "arm_id": arm_id,
        "mean_cov_z_harm_a": float(np.mean([r["cov_z_harm_a"] for r in rows])),
        "mean_level_z_harm_a": float(np.mean([r["mean_z_harm_a"] for r in rows])),
        "mean_level_z_harm_s": float(np.mean([r["mean_z_harm_s"] for r in rows])),
        "per_seed_z_harm_s": [r["mean_z_harm_s"] for r in rows],
        "min_eval_steps": int(min(r["n_eval_steps"] for r in rows)),
        "n_seeds": len(rows),
        "mean_episode_survival_steps": float(np.mean(all_survival)) if all_survival else 0.0,
        "min_episode_survival_steps": int(min(all_survival)) if all_survival else 0,
        "per_seed_mean_episode_survival_steps": [r["mean_episode_survival_steps"] for r in rows],
    }


def run(seeds=None, dry_run=False):
    if seeds is None:
        seeds = SEEDS
    import time
    t0 = time.perf_counter()

    print(f"[V3-EXQ-857a] Q-086 gentler-env confound control REDESIGN\n"
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
    benign_rows = [r for r in arm_results if r["arm_id"] == "ARM_BENIGN"]
    harsh  = _arm_agg("ARM_HARSH", harsh_rows)
    gentle = _arm_agg("ARM_GENTLE", gentle_rows)
    benign = _arm_agg("ARM_BENIGN", benign_rows)

    # --- readiness precondition 1: did the STARKEST gradient (harsh vs benign) --
    # actually change z_harm_s? This is the load-bearing gate for this redesign.
    s_harsh  = harsh["mean_level_z_harm_s"]
    s_benign = benign["mean_level_z_harm_s"]
    delta_s_starkest  = abs(s_harsh - s_benign)
    pooled_s_starkest = harsh["per_seed_z_harm_s"] + benign["per_seed_z_harm_s"]
    seed_noise_sd_starkest = max(float(np.std(pooled_s_starkest)), 1e-6)
    z_harm_s_snr_starkest = delta_s_starkest / seed_noise_sd_starkest
    manipulation_took = z_harm_s_snr_starkest > Z_HARM_S_SNR_K

    # --- comparability readout: 857's original pair (harsh vs gentle), reported ---
    # but NOT load-bearing -- the starkest pair is where the self-route is keyed.
    s_gentle = gentle["mean_level_z_harm_s"]
    delta_s_original  = abs(s_harsh - s_gentle)
    pooled_s_original = harsh["per_seed_z_harm_s"] + gentle["per_seed_z_harm_s"]
    seed_noise_sd_original = max(float(np.std(pooled_s_original)), 1e-6)
    z_harm_s_snr_original = delta_s_original / seed_noise_sd_original

    # --- readiness precondition 2 (NEW, non-gating): episode survival time -------
    surv_harsh  = harsh["mean_episode_survival_steps"]
    surv_benign = benign["mean_episode_survival_steps"]
    delta_surv  = abs(surv_harsh - surv_benign)
    pooled_surv = (harsh["per_seed_mean_episode_survival_steps"]
                   + benign["per_seed_mean_episode_survival_steps"])
    seed_noise_sd_surv = max(float(np.std(pooled_surv)), 1e-6)
    survival_snr = delta_surv / seed_noise_sd_surv
    survival_differs = survival_snr > Z_HARM_S_SNR_K

    steps_ok = (harsh["min_eval_steps"] >= (10 if dry_run else MIN_EVAL_STEPS)
                and gentle["min_eval_steps"] >= (10 if dry_run else MIN_EVAL_STEPS)
                and benign["min_eval_steps"] >= (10 if dry_run else MIN_EVAL_STEPS))

    # --- discrimination (only meaningful once manipulation_took), keyed off the --
    # starkest pair (harsh vs benign), per the autopsy's routing.
    cov_benign   = benign["mean_cov_z_harm_a"]
    level_harsh  = harsh["mean_level_z_harm_a"]
    level_benign = benign["mean_level_z_harm_a"]
    benign_saturated = cov_benign < COV_FLOOR
    level_drop_frac  = ((level_harsh - level_benign) / level_harsh
                        if level_harsh > 1e-9 else 0.0)
    level_tracks = level_drop_frac >= LEVEL_DROP_FRAC
    cov_tracks   = cov_benign >= COV_FLOOR

    calibration_pathology = bool(benign_saturated and not level_tracks)
    faithful_ecological   = bool(level_tracks or cov_tracks)

    non_degenerate = True
    degeneracy_reason = None
    if not manipulation_took:
        non_degenerate = False
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        degeneracy_reason = (
            f"Gentler-env manipulation did not take even at the STARKEST gradient "
            f"(num_hazards 4 vs 0): cross-arm z_harm_s SNR {z_harm_s_snr_starkest:.3f} "
            f"<= K={Z_HARM_S_SNR_K} (|delta|={delta_s_starkest:.4f}, "
            f"seed_noise_sd={seed_noise_sd_starkest:.4f}), despite n={len(seeds)} seeds "
            f"(double 857's n=3) and a fully hazard-free floor arm. This is a stronger "
            f"signal than 857's own inconclusive result: it suggests the sensory-tier "
            f"z_harm_s readout may not be ecologically sensitive to hazard density at "
            f"this raw-warmup scale, distinct from '857 was simply underpowered'. "
            f"857's original (harsh vs gentle) pair SNR was {z_harm_s_snr_original:.3f} "
            f"for comparison. Episode-survival-time secondary check: SNR="
            f"{survival_snr:.3f}, differs={survival_differs} -- "
            + ("also did not move." if not survival_differs else
               "DID move even though z_harm_s did not -- worth investigating as an "
               "alternate, behaviourally-grounded readout for this manipulation.")
        )
    elif not steps_ok:
        non_degenerate = False
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
        degeneracy_reason = (
            f"Insufficient eval steps for a non-degenerate readout "
            f"(harsh min {harsh['min_eval_steps']}, gentle min {gentle['min_eval_steps']}, "
            f"benign min {benign['min_eval_steps']}, floor {MIN_EVAL_STEPS})."
        )
    else:
        # Clean partition: exactly one of the two hypotheses holds.
        if calibration_pathology:
            label = "q086_calibration_pathology_representational"
        else:
            label = "q086_faithful_chronic_suffering_ecological"
        outcome = "PASS"

    # readiness-kind preconditions (recomputable: measured SNR vs threshold K).
    preconditions = [
        {
            "name": "gentle_env_manipulation_took_z_harm_s_differs_starkest",
            "description": ("cross-arm |delta mean z_harm_s| (ARM_HARSH vs ARM_BENIGN, "
                            "the starkest num_hazards 4-vs-0 gradient) as an SNR against "
                            "pooled cross-seed SD must exceed Z_HARM_S_SNR_K"),
            "measured": float(z_harm_s_snr_starkest),
            "threshold": float(Z_HARM_S_SNR_K),
            "direction": "lower",
            "control": ("ARM_HARSH vs ARM_BENIGN differ only in num_hazards/"
                        "hazard_food_attraction (4/0.7 vs 0/0.0); a real gentling must "
                        "move the sensory harm tier z_harm_s beyond seed noise"),
            "met": bool(manipulation_took),
        },
        {
            "name": "episode_survival_time_differs_starkest",
            "description": ("NON-GATING secondary check: cross-arm |delta mean "
                            "episode_survival_steps| (ARM_HARSH vs ARM_BENIGN) as an SNR "
                            "against pooled cross-seed SD, exceeding Z_HARM_S_SNR_K"),
            "measured": float(survival_snr),
            "threshold": float(Z_HARM_S_SNR_K),
            "direction": "lower",
            "control": ("Same harsh-vs-benign pair as the primary precondition; a real "
                        "hazard-density difference should also show up in how long "
                        "episodes survive before termination"),
            "met": bool(survival_differs),
            "load_bearing": False,
        },
    ]

    criteria_non_degenerate = {
        "z_harm_s_manipulation_took_starkest": bool(manipulation_took),
        "sufficient_eval_steps_all_arms": bool(steps_ok),
    }

    metrics = {
        "n_seeds": float(len(seeds)),
        "harsh_mean_cov_z_harm_a": harsh["mean_cov_z_harm_a"],
        "gentle_mean_cov_z_harm_a": gentle["mean_cov_z_harm_a"],
        "benign_mean_cov_z_harm_a": cov_benign,
        "harsh_mean_level_z_harm_a": level_harsh,
        "gentle_mean_level_z_harm_a": gentle["mean_level_z_harm_a"],
        "benign_mean_level_z_harm_a": level_benign,
        "harsh_mean_level_z_harm_s": s_harsh,
        "gentle_mean_level_z_harm_s": s_gentle,
        "benign_mean_level_z_harm_s": s_benign,
        "z_harm_s_delta_starkest": delta_s_starkest,
        "z_harm_s_snr_starkest": z_harm_s_snr_starkest,
        "z_harm_s_delta_original_pair": delta_s_original,
        "z_harm_s_snr_original_pair": z_harm_s_snr_original,
        "level_drop_frac": level_drop_frac,
        "cov_floor": COV_FLOOR,
        "manipulation_took": 1.0 if manipulation_took else 0.0,
        "calibration_pathology": 1.0 if (non_degenerate and calibration_pathology) else 0.0,
        "faithful_ecological": 1.0 if (non_degenerate and faithful_ecological) else 0.0,
        "harsh_mean_episode_survival_steps": surv_harsh,
        "gentle_mean_episode_survival_steps": gentle["mean_episode_survival_steps"],
        "benign_mean_episode_survival_steps": surv_benign,
        "episode_survival_snr_starkest": survival_snr,
        "episode_survival_differs_starkest": 1.0 if survival_differs else 0.0,
    }

    interpretation = {
        "label": label,
        "arm_harsh_summary": harsh,
        "arm_gentle_summary": gentle,
        "arm_benign_summary": benign,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "criteria": [
            {"name": "z_harm_s_manipulation_took_starkest", "load_bearing": True,
             "passed": bool(manipulation_took and steps_ok)},
        ],
        "discrimination": {
            "benign_saturated_sub_floor_cov": bool(benign_saturated),
            "level_tracks_hazard_density": bool(level_tracks),
            "cov_tracks_hazard_density": bool(cov_tracks),
            "calibration_pathology_representational": bool(non_degenerate and calibration_pathology),
            "faithful_chronic_suffering_ecological": bool(non_degenerate and faithful_ecological),
        },
        "note": (
            "Diagnostic confound control -- REDESIGN of V3-EXQ-857 per its confirmed "
            "failure autopsy's routing (more seeds, starker gradient, survival-time DV). "
            "The self-route keys off the STARKEST pair (ARM_HARSH vs ARM_BENIGN, "
            "num_hazards 4 vs 0), not 857's original (4 vs 1) pair, which is reported "
            "for comparability only. Once the gentler-env manipulation is confirmed to "
            "have moved the sensory tier z_harm_s (readiness precondition), the z_harm_a "
            "readout partitions cleanly: a still-saturated, still-high-level benign arm "
            "=> representational/calibration pathology; a level/range that tracks hazard "
            "density => faithful chronic suffering. Self-routes "
            "substrate_not_ready_requeue if z_harm_s did not move even at the starkest "
            "gradient -- if that happens, the episode-survival-time secondary check is "
            "reported explicitly rather than silently folded into the primary verdict, "
            "since a movement there while z_harm_s stays flat would itself be "
            "informative. Scoped to raw-warmup agents."
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
        "supersedes": None,
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
            "benign_env": BENIGN_ENV,
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
