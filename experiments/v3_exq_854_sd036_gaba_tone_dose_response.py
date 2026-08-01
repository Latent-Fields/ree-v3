"""V3-EXQ-854: SD-036 GABAergic decay regulator -- gaba_tone dose-response.

Validates SD-036's registered falsifier (design doc "Predicted observables" #2)
on the substrate that can finally express it. SD-036 is `candidate` /
`v3_pending` with exp_conf 0.0 and ZERO experimental entries; this is the first
run that can bear on it.

WHY THIS WAS NOT QUEUED UNTIL NOW (read before changing any DV)
---------------------------------------------------------------
SD-036's mechanism is autoregressive -- z_s(t+1) = z_s(t) * exp(-tau_s * tone)
-- but from 2026-04-22 until 2026-07-31 the wiring was not: `LatentStack.encode()`
produced z_harm (SD-010) and z_harm_a (SD-011) as PURE FEEDFORWARD encodes of the
current observation, so the regulator's rescale was discarded every tick and
z_s(t+1) was not a function of z_s(t) at all. The regulator degenerated to a
one-step CONSTANT RESCALE.

That is a DV-symmetry failure, and it is the reason this experiment must not be
written naively. The pre-registered DV `harm_norm_sustain_ratio` (= mean/peak) is
SCALE-FREE, so it was EXACTLY invariant to the manipulation: measured spread
across the whole {0.3, 0.5, 1.0, 1.5, 2.0} sweep was 8.6e-08 -- a structural
null, not a small effect. Worse, a fixed-ABSOLUTE-threshold DV would have shown a
clean monotone "dose-response" on that broken substrate: a confident-but-wrong
confirmation of a trivial rescale. Recording the correct verdict here depends on
keeping the DV scale-free AND proving non-invariance before reading it.

ree-v3 `35e8969` (2026-08-01T00:03Z) gave the harm streams a `prev_state` blend,
composing with the regulator into a leaky integrator with pole
(1 - alpha_s) * exp(-tau_s * gaba_tone). gaba_tone now moves the relaxation time
constant AND the steady-state gain -- the trajectory SHAPE, not just its scale.
Measured sustain-ratio spread post-fix: 1.4e-03 (z_harm), 1.0e-01 (z_harm_a).

DV-SYMMETRY DECLARATION (mandatory, per arm)
--------------------------------------------
* Primary DV `harm_a_sustain_ratio` = mean/peak of ||z_harm_a||. Its symmetry
  group is POSITIVE SCALAR RESCALING of the trajectory (it is a ratio of one
  arm's own quantities). The ARM_ON manipulation (gaba_tone) is NOT invariant
  under that group on this substrate: it changes the leaky-integrator pole, so
  it alters the relative shape of the trajectory, not its scale. This is
  measured, not assumed -- the P0 readiness control below re-establishes it on a
  fixed observation tape before any verdict is read, and pre-fix the same
  measurement returned ~1e-08 (exact invariance).
* ARM_OFF manipulation is the MASTER SWITCH (structural presence/absence of the
  regulator and the recurrence), not a rescale, so it is likewise not invariant.
* The DV is also invariant under time-permutation (mean and peak are both
  symmetric functions). The manipulation is a temporal decay, which changes the
  MULTISET of trajectory values, not merely their order -- so it is not
  invariant under that group either.

THE CONTROL ARM IS `use_gabaergic_decay=False`, NOT `gaba_tone=0.0`
-------------------------------------------------------------------
Post-fix, tone=0.0 suspends the DECAY but leaves the RECURRENCE active, so it is
no longer bit-equal to the master switch being off. Using it as the control would
compare two stateful substrates and credit the recurrence to the regulator.

MECH-279 IS HELD OFF
--------------------
`use_pag_freeze_gate=False` throughout. MECH-279 consumes gaba_tone as a DIRECT
SCALAR (exit_threshold = theta_freeze * gaba_tone) and never through the decay
path, so an active freeze gate would give the tone sweep a second, non-decay
route to the readout and confound SD-036 with MECH-279 (whose V3-EXQ-776 PASS
stands independently). MECH-279 is therefore NOT tagged in claim_ids.

OBSERVABLE #1 IS NOT LOAD-BEARING, AND THE ENCODER FLOOR IS THE STATED NULL
---------------------------------------------------------------------------
The design doc's observable #1 ("mode flip by ~t=50 when z_harm_norm decays below
threshold") may be UNREACHABLE at default tau/alpha, and this experiment says so
up front rather than discovering it. `harm_encoder(zeros)` has norm ~0.462 and
`affective_harm_encoder(zeros, zeros)` ~0.332, so the origin exemplar's
"z_harm_norm pinned at ~0.7 despite zero harm input" is SUBSTANTIALLY an encoder
floor response to the ambient hazard field, not purely the missing decay. Decay
fights that floor to an EQUILIBRIUM rather than returning the stream to baseline.
So: the floor is measured and recorded every run, C3 is phrased as a relative
ON-vs-OFF separation rather than a return-to-baseline, and a failure of #1 as
literally worded is reported as the floor null -- NOT as a refutation of SD-036.

CRITERIA (pre-registered; thresholds are constants below, never post-hoc)
------------------------------------------------------------------------
  C1 (LOAD-BEARING) -- the falsifier. Per-seed Spearman rho between gaba_tone and
     `harm_a_sustain_ratio` across the 5 tones is <= C1_RHO_MAX (monotone
     DECREASING: more tone -> faster decay -> lower mean/peak), AND the per-seed
     spread across the sweep is >= C1_SPREAD_FLOOR. PASS needs >= 2 of 3 seeds.
  C2 -- multi-stream cluster (observable #3). At tone 0.3 vs tone 2.0, ALL THREE
     registered streams (z_harm, z_harm_a, z_beta) show a higher sustain ratio,
     in >= 2 of 3 seeds. This is what discriminates a regulator LAYER from
     per-stream decay, which is its entire purpose.
  C3 -- ON-vs-OFF separation (observable #1, floor-robust form). The trained ON
     agent at tone=1.0 has a LOWER harm_a sustain ratio than the trained OFF
     agent, in >= 2 of 3 seeds.

Overall PASS requires C1. C2/C3 are recorded and reported but do not carry the
verdict, so a null on either is informative rather than fatal.

NULLS, DECLARED
---------------
  * C1 null (rho ~ 0 or spread below floor, with the readiness control GREEN):
    the regulator has temporal authority but no dose-response -- weakens SD-036.
  * C1 spread below VACUITY_CEILING: NOT a scientific null. That is the pre-fix
    signature returning, so the run self-routes `substrate_not_ready_requeue`
    and sets non_degenerate=false rather than reporting a refutation.
  * C2 null: decay is real but stream coverage is narrower than the regulator-
    layer commitment predicts -- routes to the per-stream alternative, not to
    "no decay".
  * C3 null: consistent with the encoder floor dominating the equilibrium; does
    NOT bear on C1.

No sleep machinery is used, so no SLEEP DRIVER line applies.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_EXPERIMENTS_DIR = _THIS.parent
_REE_V3_ROOT = _THIS.parents[1]
for _p in (str(_EXPERIMENTS_DIR), str(_REE_V3_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_protocol import emit_outcome  # noqa: E402

from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.stats import spearman  # noqa: E402
from _lib.baselines import sd036_decay as B  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from pack_writer import write_flat_manifest  # noqa: E402

# Agents are built per-cell and discarded, so the counters are accumulated as we
# go rather than by holding every arm x seed agent alive to the end of the run.
_ZG = ZGoalStreamAccumulator()


EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_854_sd036_gaba_tone_dose_response"
CLAIM_IDS = ["SD-036"]
QUEUE_ID = "V3-EXQ-854"

# --- the registered sweep (design doc observable #2) ---
TONE_SWEEP: List[float] = [0.3, 0.5, 1.0, 1.5, 2.0]
BASELINE_TONE = 1.0

ARM_OFF = "decay_off_legacy"
ARM_ON = "decay_on"
ARMS = [ARM_OFF, ARM_ON]

# --- pre-registered thresholds (NEVER derived from the run's own statistics) ---
C1_RHO_MAX = -0.9          # monotone decreasing across the 5 tones
C1_SPREAD_FLOOR = 1e-2     # measured post-fix spread for z_harm_a was ~1.0e-01
C2_SEEDS_REQUIRED = 2
C1_SEEDS_REQUIRED = 2
C3_SEEDS_REQUIRED = 2

# The pre-fix substrate measured 8.6e-08 on this exact DV. Anything at or below
# this ceiling means the manipulation did nothing but rescale -- i.e. the defect
# is back. This is a SUBSTRATE-READINESS floor, not a scientific threshold.
VACUITY_CEILING = 1e-5
# Readiness control floor: comfortably above vacuity, far below the measured
# post-fix effect (~7e-02 shape deviation on the contract tape).
READINESS_SPREAD_FLOOR = 1e-3

READINESS_TAPE_STEPS = 60


def _batch(v: Any) -> Optional[torch.Tensor]:
    if v is None:
        return None
    v = v.float()
    return v.unsqueeze(0) if v.dim() == 1 else v


def _record_tape(seed: int, steps: int) -> List[Dict[str, Any]]:
    """A fixed observation tape, independent of any agent.

    Actions come from a dedicated RandomState rather than from an agent, so the
    identical tape is replayed into every tone and the arms cannot drift apart
    through their own action choices. This is the contract-suite methodology
    (tests/contracts/test_sd_036_gabaergic_decay.py C11-C16).
    """
    env = B.make_env(seed)
    _, od = env.reset()
    rng = np.random.RandomState(seed)
    frames: List[Dict[str, Any]] = []
    for _ in range(steps):
        frames.append(
            dict(
                body=_batch(od["body_state"]),
                world=_batch(od["world_state"]),
                harm=_batch(od.get("harm_obs")),
                harm_a=_batch(od.get("harm_obs_a")),
                hist=_batch(od.get("harm_history")),
            )
        )
        _, od, _, done, _ = env.step(int(rng.randint(env.action_dim)))
        if done:
            _, od = env.reset()
    return frames


def _replay_tape_sustain(seed: int, frames: List[Dict[str, Any]], tone: float) -> float:
    """Replay the fixed tape into a FRESH ON agent at `tone`; return harm_a mean/peak."""
    env = B.make_env(seed)
    _, od0 = env.reset()
    agent = B.make_agent(env, od0, use_gabaergic_decay=True, gaba_tone=tone)
    agent.reset()
    agent.eval()
    norms: List[float] = []
    with torch.no_grad():
        for fr in frames:
            lat = agent.sense(
                fr["body"],
                fr["world"],
                obs_harm=fr["harm"],
                obs_harm_a=fr["harm_a"],
                obs_harm_history=fr["hist"],
            )
            z = getattr(lat, "z_harm_a", None)
            norms.append(float(z.norm()) if z is not None else float("nan"))
    return B.sustain_ratio(np.asarray(norms, dtype=np.float64))


def readiness_control(seed: int) -> Dict[str, Any]:
    """P0 POSITIVE CONTROL -- does the instrument register a tone effect at all?

    Asserts the SAME STATISTIC the load-bearing criterion C1 routes on (the
    SPREAD of harm_a_sustain_ratio across the tone sweep), on a condition where
    the effect is known to exist (fixed observation tape, fresh agents -- the
    contract-suite C14 setup). This is deliberately NOT a magnitude proxy: a
    magnitude readiness check would pass on a uniform rescale while the
    range-gated criterion was starved (the V3-EXQ-643 defect).

    Below the floor means SUBSTRATE NOT READY (the pre-fix feedforward wiring is
    back), never "SD-036 is refuted".
    """
    frames = _record_tape(seed, READINESS_TAPE_STEPS)
    ratios = [_replay_tape_sustain(seed, frames, t) for t in TONE_SWEEP]
    spread = float(np.nanmax(ratios) - np.nanmin(ratios))
    return {
        "seed": seed,
        "tones": list(TONE_SWEEP),
        "sustain_ratios": [float(r) for r in ratios],
        "spread": spread,
    }


def run_cell(arm: str, seed: int) -> Dict[str, Any]:
    """One (seed x arm) cell: train, then evaluate."""
    print(f"Seed {seed} Condition {arm}", flush=True)

    use_decay = arm == ARM_ON
    # The OFF arm is the canonical baseline -> mint it reuse-eligible with the
    # driver EXCLUDED from the hash, so a later sibling with a different driver
    # can match it (arm_reuse_fingerprint_plan.md 9.4/9.7).
    if arm == ARM_OFF:
        config_slice = B.off_path_config_slice()
        include_driver = False
    else:
        config_slice = dict(B.off_path_config_slice())
        config_slice["off_arm_flags"] = {
            "use_gabaergic_decay": True,
            "gaba_tone_train": BASELINE_TONE,
        }
        config_slice["tone_sweep"] = list(TONE_SWEEP)
        include_driver = True

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=_THIS,
        config_slice_declared=True,
        include_driver_script_in_hash=include_driver,
    ) as cell:
        env = B.make_env(seed)
        _, od0 = env.reset()
        agent = B.make_agent(
            env, od0, use_gabaergic_decay=use_decay, gaba_tone=BASELINE_TONE
        )

        floor = B.encoder_floor_norms(agent)
        train_diag = B.train_agent(agent, env, label=f"{arm} seed={seed}", seed=seed)

        # ---- P2: frozen evaluation. No learning, so the SAME trained agent can
        # be re-evaluated at several tones without earlier evaluations changing
        # its weights. Each tone gets a fresh env at the same seed.
        per_tone: List[Dict[str, Any]] = []
        tones = TONE_SWEEP if use_decay else [None]
        for tone in tones:
            eval_env = B.make_env(seed)
            if use_decay and agent.gabaergic_decay is not None:
                agent.gabaergic_decay.set_gaba_tone(float(tone))
            traj = B.record_stream_trajectories(agent, eval_env, steps=B.EVAL_STEPS)
            tone_row = {
                "gaba_tone": (float(tone) if tone is not None else None),
                "harm_a_sustain_ratio": B.sustain_ratio(traj["z_harm_a"]),
                "harm_sustain_ratio": B.sustain_ratio(traj["z_harm"]),
                "beta_sustain_ratio": B.sustain_ratio(traj["z_beta"]),
                "harm_a_peak": float(np.nanmax(traj["z_harm_a"])),
                "harm_a_mean": float(np.nanmean(traj["z_harm_a"])),
                "harm_peak": float(np.nanmax(traj["z_harm"])),
                "beta_peak": float(np.nanmax(traj["z_beta"])),
                # Record the full trajectories: they are the raw material for
                # any successor question and cost almost nothing to bank.
                "z_harm_a_trajectory": [float(x) for x in traj["z_harm_a"]],
                "z_harm_trajectory": [float(x) for x in traj["z_harm"]],
                "z_beta_trajectory": [float(x) for x in traj["z_beta"]],
                # Observable #1's BEHAVIOURAL readout, under the verbatim
                # V3-EXQ-475 / EXQ-471 classifier. Non-gating -- see
                # floor_vs_mode_threshold for why a 100% avoid_frac may be an
                # encoder-floor property rather than an SD-036 failure.
                "mode_metrics": B.mode_metrics(traj["modes"]),
                "mode_sequence": list(traj["modes"]),
            }
            per_tone.append(tone_row)

        # Shape deviation vs the tone=1.0 reference (scale-free, so it is the
        # readout that was exactly zero pre-fix).
        if use_decay:
            ref = np.asarray(
                next(
                    r["z_harm_a_trajectory"]
                    for r in per_tone
                    if r["gaba_tone"] == BASELINE_TONE
                ),
                dtype=np.float64,
            )
            for r in per_tone:
                r["harm_a_shape_deviation_vs_tone1"] = B.shape_deviation(
                    np.asarray(r["z_harm_a_trajectory"], dtype=np.float64), ref
                )

        row: Dict[str, Any] = {
            "arm_id": arm,
            "seed": seed,
            "use_gabaergic_decay": use_decay,
            "per_tone": per_tone,
            "encoder_floor": floor,
            "train_diagnostics": train_diag,
        }
        cell.stamp(row)
        # AFTER stepping -- the accumulator reads the counters at call time.
        _ZG.observe(agent)

    # Per-cell verdict line (the runner counts these; seeds x conditions of them).
    if use_decay:
        ratios = [r["harm_a_sustain_ratio"] for r in per_tone]
        rho = spearman(TONE_SWEEP, ratios)
        spread = float(np.nanmax(ratios) - np.nanmin(ratios))
        row["cell_rho"] = rho
        row["cell_spread"] = spread
        cell_pass = (
            rho is not None and rho <= C1_RHO_MAX and spread >= C1_SPREAD_FLOOR
        )
    else:
        # The control arm has no dose-response to pass. Its verdict reports
        # whether it produced a usable baseline measurement at all, so a broken
        # control is visible in the runner log rather than silently green.
        row["cell_rho"] = None
        row["cell_spread"] = None
        cell_pass = bool(
            per_tone and np.isfinite(per_tone[0]["harm_a_sustain_ratio"])
            and per_tone[0]["harm_a_peak"] > 0.0
        )

    row["cell_pass"] = bool(cell_pass)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    return row


def evaluate(arm_results: List[Dict[str, Any]], readiness: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply the pre-registered criteria."""
    on_rows = [r for r in arm_results if r["arm_id"] == ARM_ON]
    off_rows = {r["seed"]: r for r in arm_results if r["arm_id"] == ARM_OFF}

    # ---- C1: dose-response (LOAD-BEARING) ----
    c1_per_seed = []
    for r in on_rows:
        ratios = [t["harm_a_sustain_ratio"] for t in r["per_tone"]]
        rho = spearman(TONE_SWEEP, ratios)
        spread = float(np.nanmax(ratios) - np.nanmin(ratios))
        c1_per_seed.append(
            {
                "seed": r["seed"],
                "rho": rho,
                "spread": spread,
                "passed": bool(
                    rho is not None and rho <= C1_RHO_MAX and spread >= C1_SPREAD_FLOOR
                ),
            }
        )
    c1_n = sum(1 for s in c1_per_seed if s["passed"])
    c1_passed = c1_n >= C1_SEEDS_REQUIRED

    # ---- C2: multi-stream cluster at tone 0.3 vs 2.0 ----
    c2_per_seed = []
    for r in on_rows:
        by_tone = {t["gaba_tone"]: t for t in r["per_tone"]}
        lo, hi = by_tone.get(0.3), by_tone.get(2.0)
        if lo is None or hi is None:
            c2_per_seed.append({"seed": r["seed"], "passed": False, "streams": {}})
            continue
        streams = {
            "z_harm": lo["harm_sustain_ratio"] > hi["harm_sustain_ratio"],
            "z_harm_a": lo["harm_a_sustain_ratio"] > hi["harm_a_sustain_ratio"],
            "z_beta": lo["beta_sustain_ratio"] > hi["beta_sustain_ratio"],
        }
        c2_per_seed.append(
            {"seed": r["seed"], "passed": all(streams.values()), "streams": streams}
        )
    c2_n = sum(1 for s in c2_per_seed if s["passed"])
    c2_passed = c2_n >= C2_SEEDS_REQUIRED

    # ---- C3: ON-vs-OFF separation at tone 1.0 (floor-robust form of #1) ----
    c3_per_seed = []
    for r in on_rows:
        off = off_rows.get(r["seed"])
        by_tone = {t["gaba_tone"]: t for t in r["per_tone"]}
        on_t1 = by_tone.get(BASELINE_TONE)
        if off is None or on_t1 is None or not off["per_tone"]:
            c3_per_seed.append({"seed": r["seed"], "passed": False})
            continue
        off_ratio = off["per_tone"][0]["harm_a_sustain_ratio"]
        c3_per_seed.append(
            {
                "seed": r["seed"],
                "on_tone1": on_t1["harm_a_sustain_ratio"],
                "off": off_ratio,
                "passed": bool(on_t1["harm_a_sustain_ratio"] < off_ratio),
            }
        )
    c3_n = sum(1 for s in c3_per_seed if s["passed"])
    c3_passed = c3_n >= C3_SEEDS_REQUIRED

    # ---- readiness / degeneracy ----
    control_spreads = [k["spread"] for k in readiness]
    min_control_spread = float(np.nanmin(control_spreads)) if control_spreads else 0.0
    observed_spreads = [s["spread"] for s in c1_per_seed]
    max_observed_spread = float(np.nanmax(observed_spreads)) if observed_spreads else 0.0

    ready = min_control_spread >= READINESS_SPREAD_FLOOR
    vacuous = max_observed_spread <= VACUITY_CEILING

    return {
        "C1_dose_response": {
            "load_bearing": True,
            "passed": c1_passed,
            "seeds_passing": c1_n,
            "seeds_required": C1_SEEDS_REQUIRED,
            "per_seed": c1_per_seed,
            "rho_max": C1_RHO_MAX,
            "spread_floor": C1_SPREAD_FLOOR,
        },
        "C2_multi_stream_cluster": {
            "load_bearing": False,
            "passed": c2_passed,
            "seeds_passing": c2_n,
            "per_seed": c2_per_seed,
        },
        "C3_on_vs_off_separation": {
            "load_bearing": False,
            "passed": c3_passed,
            "seeds_passing": c3_n,
            "per_seed": c3_per_seed,
        },
        "_ready": ready,
        "_vacuous": vacuous,
        "_min_control_spread": min_control_spread,
        "_max_observed_spread": max_observed_spread,
    }


def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    print("[P0] readiness positive control (fixed tape, fresh agents)", flush=True)
    readiness = [readiness_control(s) for s in seeds]
    for k in readiness:
        print(
            f"  [P0] seed={k['seed']} control tone-spread={k['spread']:.3e} "
            f"(floor {READINESS_SPREAD_FLOOR:.1e})",
            flush=True,
        )

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            arm_results.append(run_cell(arm, seed))

    ev = evaluate(arm_results, readiness)
    ready = ev.pop("_ready")
    vacuous = ev.pop("_vacuous")
    min_control_spread = ev.pop("_min_control_spread")
    max_observed_spread = ev.pop("_max_observed_spread")

    # --- routing ---------------------------------------------------------
    # A below-floor control, or an observed spread at the pre-fix vacuity level,
    # is a SUBSTRATE-READINESS failure. It self-routes to requeue and is marked
    # degenerate so governance does not score it -- it is NEVER reported as a
    # refutation of SD-036.
    substrate_not_ready = (not ready) or vacuous
    if substrate_not_ready:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = (
            f"harm_a sustain-ratio tone-spread is at the pre-fix vacuity level "
            f"(control min {min_control_spread:.3e} < floor "
            f"{READINESS_SPREAD_FLOOR:.1e}, or observed max "
            f"{max_observed_spread:.3e} <= ceiling {VACUITY_CEILING:.1e}). The "
            f"decay regulator has no temporal authority over z_harm_a on this "
            f"substrate, so the dose-response DV is structurally invariant. This "
            f"is the 2026-04-22..2026-07-31 feedforward defect, not a null result."
        )
    else:
        c1 = ev["C1_dose_response"]["passed"]
        outcome = "PASS" if c1 else "FAIL"
        label = "sd036_dose_response_confirmed" if c1 else "sd036_dose_response_absent"
        evidence_direction = "supports" if c1 else "weakens"
        non_degenerate = True
        degeneracy_reason = ""

    elapsed = time.perf_counter() - t0
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    # Encoder floor, aggregated -- the stated null for observable #1.
    floors = [r["encoder_floor"] for r in arm_results if r.get("encoder_floor")]
    floor_summary: Dict[str, float] = {}
    if floors:
        for key in floors[0]:
            floor_summary[key] = float(np.mean([f[key] for f in floors if key in f]))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "acceptance": ev,
        "arm_results": arm_results,
        "readiness_control": readiness,
        "encoder_floor_summary": floor_summary,
        # Quantifies the observable-#1 caveat instead of only asserting it: if
        # the ambient encoder floor already exceeds the avoid threshold, the
        # mode lock cannot be resolved by ANY decay rate, and a 100% avoid_frac
        # is not evidence against SD-036.
        "observable_1_reachability": B.floor_vs_mode_threshold(floor_summary),
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "harm_a_sustain_ratio_tone_spread_control",
                    "description": (
                        "Spread of harm_a_sustain_ratio across the gaba_tone sweep on "
                        "a fixed observation tape with fresh agents (positive control). "
                        "This is the SAME statistic C1 routes on -- deliberately not a "
                        "magnitude proxy."
                    ),
                    "control": "fixed 60-step observation tape, fresh ON agents per tone",
                    "measured": min_control_spread,
                    "threshold": READINESS_SPREAD_FLOOR,
                    # FLOOR: met when measured >= threshold. Stated explicitly so
                    # the indexer's recompute matches the author's `met` rather
                    # than defaulting (the 648a/649 directionality bug).
                    "direction": "lower",
                    "comparator": ">=",
                    "met": bool(ready),
                }
            ],
            "criteria_non_degenerate": {
                "C1_dose_response": bool(not vacuous),
                "C2_multi_stream_cluster": bool(not vacuous),
                "C3_on_vs_off_separation": bool(not vacuous),
            },
            "criteria": [
                {"name": "C1_dose_response", "load_bearing": True,
                 "passed": bool(ev["C1_dose_response"]["passed"])},
                {"name": "C2_multi_stream_cluster", "load_bearing": False,
                 "passed": bool(ev["C2_multi_stream_cluster"]["passed"])},
                {"name": "C3_on_vs_off_separation", "load_bearing": False,
                 "passed": bool(ev["C3_on_vs_off_separation"]["passed"])},
            ],
            "observable_1_note": (
                "Observable #1 as literally worded (mode flip by ~t=50 on return to "
                "baseline) is NOT load-bearing here. harm_encoder(zeros) and "
                "affective_harm_encoder(zeros, zeros) have non-zero norm, so the "
                "V3-EXQ-471 lock is substantially an encoder FLOOR response to the "
                "ambient hazard field; decay reaches an equilibrium against that floor "
                "rather than returning to baseline. Measured floor is recorded in "
                "encoder_floor_summary. C3 is the floor-robust relative form."
            ),
        },
        # Descriptive only -- experiment_ethics_preflight.md is a DRAFT that binds
        # at V4 and is explicitly NON-BLOCKING for V3. Recorded honestly rather
        # than as the all-false template: this experiment does drive the harm
        # streams by design. V3 has no self-model, no autobiographical memory and
        # no social mind, and the harm streams are pre-ethical instrumentation
        # under the SENT-0 boundary.
        "ethics_preflight": {
            "involves_negative_valence": True,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "notes": (
                "use_harm_suffering_accumulator=False (MECH-219 inactive, s_t cannot "
                "rise); harm_suffering_escapability_constant=1.0 (fully escapable); "
                "no sleep/replay. Peak ||z_harm_a|| is recorded per arm/tone in "
                "arm_results[].per_tone[].harm_a_peak for audit against the "
                "Section 3.2 caps."
            ),
            "decision": "allow",
        },
        "sd036_substrate_note": (
            "Requires ree-v3 35e8969 (2026-08-01) harm-stream decay recurrence. "
            "The control arm is use_gabaergic_decay=False, NOT gaba_tone=0.0 -- "
            "post-fix, tone 0.0 suspends decay but leaves the recurrence active."
        ),
    }
    if degeneracy_reason:
        manifest["degeneracy_reason"] = degeneracy_reason

    full_config = {
        "env_kwargs": dict(B.ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": B.P0_WARMUP_EPISODES,
            "p1_main_episodes": B.P1_MAIN_EPISODES,
            "total_train_episodes": B.TOTAL_TRAIN_EPISODES,
            "steps_per_episode": B.STEPS_PER_EPISODE,
            "eval_steps": B.EVAL_STEPS,
        },
        "tone_sweep": list(TONE_SWEEP),
        "baseline_tone": BASELINE_TONE,
        "arms": list(ARMS),
        "thresholds": {
            "C1_RHO_MAX": C1_RHO_MAX,
            "C1_SPREAD_FLOOR": C1_SPREAD_FLOOR,
            "C1_SEEDS_REQUIRED": C1_SEEDS_REQUIRED,
            "C2_SEEDS_REQUIRED": C2_SEEDS_REQUIRED,
            "C3_SEEDS_REQUIRED": C3_SEEDS_REQUIRED,
            "VACUITY_CEILING": VACUITY_CEILING,
            "READINESS_SPREAD_FLOOR": READINESS_SPREAD_FLOOR,
        },
        "use_pag_freeze_gate": False,
        "latent_dims": B.off_path_config_slice()["latent_dims"],
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=list(seeds),
        script_path=_THIS,
        started_at=t0,
        elapsed_seconds=elapsed,
        z_goal_stream_stats=_ZG.stats(),
    )
    return {"outcome": outcome, "manifest_path": out_path, "manifest": manifest}


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="V3-EXQ-854 SD-036 gaba_tone dose-response")
    ap.add_argument("--dry-run", action="store_true", help="smoke test at toy scale")
    args = ap.parse_args()

    if args.dry_run:
        # Toy scale. Deliberately shrinks the SCHEDULE only -- never a threshold.
        B.P0_WARMUP_EPISODES = 1
        B.P1_MAIN_EPISODES = 1
        B.TOTAL_TRAIN_EPISODES = 2
        B.STEPS_PER_EPISODE = 20
        B.EVAL_STEPS = 30
        seeds = B.SEEDS[:1]
        globals()["READINESS_TAPE_STEPS"] = 20
    else:
        seeds = list(B.SEEDS)

    result = run_experiment(seeds, dry_run=args.dry_run)
    out_path = result["manifest_path"]

    print(f"outcome: {result['outcome']}", flush=True)
    print(f"manifest: {out_path}", flush=True)
    print(
        "acceptance: "
        + json.dumps(
            {
                k: v.get("passed")
                for k, v in result["manifest"]["acceptance"].items()
            }
        ),
        flush=True,
    )

    return {
        "outcome": result["outcome"],
        "manifest_path": out_path,
        "run_id": result["manifest"]["run_id"],
        "dry_run": bool(args.dry_run),
    }


if __name__ == "__main__":
    _res = main()
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res["manifest_path"],
        run_id=_res["run_id"],
        queue_id=QUEUE_ID,
        dry_run=_res["dry_run"],
    )
