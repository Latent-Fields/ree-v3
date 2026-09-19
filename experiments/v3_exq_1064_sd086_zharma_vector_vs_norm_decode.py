"""V3-EXQ-1064 -- SD-086 option C: does the 16-d z_harm_a VECTOR carry more than its NORM?

!!  REFUSED AT DESIGN 2026-09-19 -- **NOT QUEUED**. DO NOT QUEUE THIS AS-IS.  !!

RED-TEAM (Step 4.5, model: fable): **BLOCKING**. Confirmed independently by this session at the
real eval budget, 3 seeds, with and without the P0h stage:

    UNTRAINED (frozen random projection)  vector_r2 0.983-0.9995  mean margin +0.577  C1 T C2 T  PASS
    TRAINED   (P0h stage ON)              vector_r2 0.999-1.0000  mean margin +0.552  C1 T C2 T  PASS

THE DESIGN PASSES ON AN UNTRAINED ENCODER. Root cause: the scored decode targets
`hazard_at_agent` / `resource_at_agent` ARE two coordinates of the encoder's own INPUT
(`causal_grid_world.py` ~3037-3038 writes them into `harm_obs_a`, which this driver reads back at
`_collect`), and `AffectiveHarmEncoder` is a near-linear 58->64->16 map over a tiny input range,
so its 16-d output linearly reconstructs its own input for ANY weights. `vector_r2` is therefore
pinned at ~1.0, the scored margin collapses to `1 - norm_r2`, and BOTH pre-registered criteria are
decided by the norm alone -- nothing the P0h training does can reach the DV. The headroom
precondition `min(1 - norm_r2) > 0.10` then IMPLIES C2 `mean margin > 0.10`, so C2 cannot fail on
any run that is scored at all.

What it would actually measure: "16 dims beat 1 dim at reconstructing a rank-2 input" -- a
statement about RANK, true of a random projection, not about what training put into z_harm_a, and
not about the readout FORM that SD-086 is a claim about.

This file is kept because the harness, precondition machinery and telemetry are sound and a
re-specified successor should start from it -- NOT because the design is runnable. Full diagnosis
and the confirming numbers:
`REE_assembly/evidence/planning/sd086_optc_refused_at_design_staged_20260919.md`.

THE QUESTION, and why it is posed this way. SD-086 asserts that z_harm_a's functional readout
must be a calibrated scalar head rather than the latent NORM, because "the norm conflates a large
near-constant encoder offset with a small functional component". Option C
(`sd086_zharma_readout_precondition_staged_20260918.md` sec 6) operationalises that directly as a
contrast rather than as a floor:

    does a linear decode from the 16-d z_harm_a beat a decode from ||z_harm_a|| alone,
    beyond the cross-seed noise band?

The original pre-registered precondition (a decode floor) was REFUSED at design on 2026-09-18
because it clears vacuously: a random projection of an informative scalar is linearly decodable,
so "clears a floor" said nothing about SD-086's thesis. The vector-vs-norm form cannot clear that
way, because both readouts see the SAME latent on the SAME ticks -- the only difference is how
much of it the readout is allowed to use.

WHY THE CONTRAST IS NOT TRIVIALLY WON EITHER. ||z|| is a NONLINEAR function of z, so it is not in
the linear span of the 16 features. The 1-d norm readout is therefore NOT a nested sub-model of
the 16-d readout, and the margin can genuinely come out NEGATIVE on held-out data (a 1-parameter
fit can generalise better than a 16-parameter one). The measured direction is a real result, not
an algebraic identity. This is the DV-symmetry declaration the design audit requires: the
manipulation (which feature set the readout receives) is NOT invariant under any symmetry of the
DV (held-out R^2), precisely because the norm reduction is not an invertible linear map.

THE DECODE TARGET, and why THIS one. `harm_obs_a` is STRUCTURALLY RANK 2 -- the env writes
`hazard_at_agent` into dims [:25] and `resource_at_agent` into dims [25:]
(`ree_core/environment/causal_grid_world.py` ~3033-3038). The scored target is therefore the
JOINT pair [hazard_at_agent, resource_at_agent]: exactly the two degrees of freedom the substrate
has. That choice is deliberate and was the user's (2026-09-19T07:06Z, OPTION M). A single-scalar
target would measure inside the norm's reach -- one scalar is what a norm can carry -- so the
joint target is the one that uses the rank-2 ceiling rather than fighting it. It is also NOT the
most favourable candidate (see the telemetry block), so it is not cherry-picked.

THE RANK-2 LIMIT -- STATE IT WHEREVER THIS RESULT IS READ. z_harm_a can carry at most 2
independent d.o.f. ABOUT THE WORLD no matter what it is trained on; training changes WHICH two,
not how many. So a WEAK vector-over-norm margin must NOT be over-read as "the vector is nearly as
poor as the norm" -- the ceiling is low for both. Full treatment:
`REE_assembly/evidence/planning/harm_obs_a_rank2_scoping_staged_20260918.md`.

PRECONDITION -- A RUN ON AN UNREADY ENCODER SELF-ROUTES, IT DOES NOT SCORE. The contrast is only
meaningful on a TRAINED encoder. Before 2026-09-19 no driver trained the affective encoder at all
(`sd_zharm_a_warmup_optimizer_group`); the P0h stage
(`experiments/_lib/zharm_a_p0_warmup.py`, ree-v3 78397036) fixed that, and carries its own
readiness verdict `p0h_readiness_met` = (encoder tensors moved) AND (held-out aux readout beats a
constant-mean predictor). This driver ASSERTS that verdict per seed and, if any seed is unready,
writes `substrate_not_ready_requeue` and does NOT score the criteria. That is the difference
between "the vector carries nothing extra" and "the encoder was never taught anything".

PRE-REGISTERED PASS RULE (user-pinned, 2026-09-19T07:06Z; constants below, never derived
post-hoc). PASS iff BOTH:
    C1  mean margin > 2 x the cross-seed SD of the margin      (beyond the noise band)
    C2  mean margin > 0.10                                     (absolute floor)
over >= 3 seeds, with the readiness precondition met on every seed.
Combination rule: C1 AND C2. Both are load-bearing.

SELF-ROUTING. vector >> norm (PASS) justifies SD-086's trained-head build; vector ~ norm (FAIL)
weakens the premise and correctly avoids that build. Neither routes a substrate defect -- an
unready encoder is caught by the precondition first.

NOTE ON THE PER-SEED `verdict:` LINES. The runner counts one `verdict:` per seed x condition.
Those lines report whether THAT seed's own margin cleared C2's absolute floor; they are progress
instrumentation, NOT the run outcome. The manifest `outcome` is the cross-seed rule above.

MECH-094: not applicable (waking observation stream, no replay content).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.capability_eval import RandomPolicy
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments._lib.zharm_a_p0_warmup import run_zharm_a_p0
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_1064_sd086_zharma_vector_vs_norm_decode"
CLAIM_IDS = ["SD-086"]

# --- pre-registered constants (user-pinned 2026-09-19T07:06Z; never derived from the run) ---
MARGIN_NOISE_BAND_K = 2.0      # C1: mean margin must exceed K x cross-seed SD
MARGIN_ABSOLUTE_FLOOR = 0.10   # C2: mean margin must exceed this outright
MIN_SEEDS = 3

SEEDS = (0, 1, 2)
HARM_HISTORY_LEN = 8
P0H_EPISODES = 12              # training episodes per seed -- the [train] ep N/M denominator
P0H_STEPS_PER_EPISODE = 25
EVAL_TICKS = 600
EVAL_SEED_OFFSET = 100         # eval env seed = EVAL_SEED_OFFSET + train seed (held out)
RIDGE = 1e-6

# The two structural d.o.f. of harm_obs_a (causal_grid_world.py ~3033-3038): dims [:25] carry
# hazard_at_agent, dims [25:] carry resource_at_agent. HAZARD_DIMS is read from the live vector
# rather than hardcoded at 25, so a future layout change fails loudly instead of silently
# decoding the wrong halves.
SCORED_TARGET = "joint_hazard_resource"


def _split_point(harm_obs_a_dim: int) -> int:
    if harm_obs_a_dim % 2 != 0:
        raise ValueError(
            "harm_obs_a_dim=%d is not even -- the [:n] hazard / [n:] resource split this "
            "experiment decodes no longer holds; re-read causal_grid_world.py ~3033-3038"
            % (harm_obs_a_dim,)
        )
    return harm_obs_a_dim // 2


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, use_proxy_fields=True,
                             harm_history_len=HARM_HISTORY_LEN)


def _make_agent(seed: int) -> REEAgent:
    env = _make_env(seed)
    _flat, obs = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=obs["body_state"].shape[-1],
        world_obs_dim=obs["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg.latent.use_affective_harm_stream = True
    cfg.latent.harm_history_len = HARM_HISTORY_LEN
    cfg.latent.harm_obs_a_dim = obs["harm_obs_a"].shape[-1]
    # SD-020 PE target + the P0h stage's adopted precision floor (its default) -- this is the
    # configuration the 2026-09-19 option-I measurement established as the one that trains.
    cfg.harm_surprise_pe_enabled = True
    agent = REEAgent(cfg)
    # from_dims silently swallows unknown kwargs, so assert rather than assume.
    if not getattr(agent.config.latent, "use_affective_harm_stream", False):
        raise RuntimeError("use_affective_harm_stream did not take -- z_harm_a would be None")
    if not bool(getattr(agent.config, "harm_surprise_pe_enabled", False)):
        raise RuntimeError("harm_surprise_pe_enabled did not take -- wrong P0h target")
    return agent


def _holdout_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Held-out R^2 of a ridge linear fit, 50/50 split by TIME (not shuffled).

    A time split, not a random one: adjacent ticks are correlated, so a shuffled split leaks the
    target across the boundary and inflates BOTH readouts -- which would not cancel in the margin,
    because the 16-d readout has more capacity to exploit the leak.
    """
    n = len(y)
    k = n // 2
    if k < 2:
        return float("nan")
    xtr = np.c_[x[:k], np.ones(k)]
    xte = np.c_[x[k:], np.ones(n - k)]
    ytr, yte = y[:k], y[k:]
    w = np.linalg.solve(xtr.T @ xtr + RIDGE * np.eye(xtr.shape[1]), xtr.T @ ytr)
    pred = xte @ w
    ss_res = float(((yte - pred) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _margin(z: np.ndarray, norm: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Mean over target columns of (vector R^2 - norm R^2), plus both halves."""
    vec = [_holdout_r2(z, targets[:, j]) for j in range(targets.shape[1])]
    nrm = [_holdout_r2(norm, targets[:, j]) for j in range(targets.shape[1])]
    v = float(np.mean(vec))
    n = float(np.mean(nrm))
    return {"vector_r2": v, "norm_r2": n, "margin": v - n}


def _collect(agent: REEAgent, seed: int, n_ticks: int) -> Dict[str, np.ndarray]:
    """Roll out a HELD-OUT env and collect the latent plus every candidate target."""
    env = _make_env(EVAL_SEED_OFFSET + seed)
    _flat, obs = env.reset()
    agent.reset()
    policy = RandomPolicy(EVAL_SEED_OFFSET + seed)
    policy.reset(env)
    split = _split_point(int(agent.config.latent.harm_obs_a_dim))

    zs: List[np.ndarray] = []
    hazard: List[float] = []
    resource: List[float] = []
    harm_exposure: List[float] = []
    for _ in range(n_ticks):
        hoa = obs["harm_obs_a"].float()
        hh = obs["harm_history"].float()
        with torch.no_grad():
            z, _pred = agent.latent_stack.affective_harm_encoder(
                hoa.unsqueeze(0), hh.unsqueeze(0))
        zs.append(z.reshape(-1).numpy().astype(np.float64))
        hoa_np = hoa.numpy().astype(np.float64)
        hazard.append(float(hoa_np[:split].mean()))
        resource.append(float(hoa_np[split:].mean()))
        harm_exposure.append(float(obs["harm_obs"].reshape(-1)[-1]))
        action = policy.act(env, obs)
        with torch.no_grad():
            _flat, _h, done, _info, obs = env.step(action)
        if done:
            _flat, obs = env.reset()
    return {
        "z": np.asarray(zs),
        "hazard": np.asarray(hazard),
        "resource": np.asarray(resource),
        "harm_exposure": np.asarray(harm_exposure),
    }


def _config_slice(dry_run: bool) -> Dict[str, Any]:
    return {
        "env": {"use_proxy_fields": True, "harm_history_len": HARM_HISTORY_LEN},
        "latent": {"use_affective_harm_stream": True,
                   "harm_history_len": HARM_HISTORY_LEN},
        "harm_surprise_pe_enabled": True,
        "schedule": {"p0h_episodes": 2 if dry_run else P0H_EPISODES,
                     "p0h_steps_per_episode": 6 if dry_run else P0H_STEPS_PER_EPISODE,
                     "eval_ticks": 60 if dry_run else EVAL_TICKS},
        "readout": {"ridge": RIDGE, "split": "time_50_50"},
    }


def _run_seed(seed: int, dry_run: bool, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    episodes = 2 if dry_run else P0H_EPISODES
    steps = 6 if dry_run else P0H_STEPS_PER_EPISODE
    eval_ticks = 60 if dry_run else EVAL_TICKS

    print("Seed %d Condition vector_vs_norm" % seed, flush=True)
    with arm_cell(seed, config_slice=_config_slice(dry_run),
                  script_path=Path(__file__)) as cell:
        agent = _make_agent(seed)
        # P0h prints its own "[train] ... ep N/M" progress lines with `episodes` as the
        # denominator -- that is the loop bound, matching episodes_per_run in the queue entry.
        p0h = run_zharm_a_p0(agent, _make_env(seed), seed, episodes, steps,
                             RandomPolicy(seed), label="exq1064",
                             dry_run=dry_run)
        data = _collect(agent, seed, eval_ticks)
        zg.observe(agent)

        z = data["z"]
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        joint = np.c_[data["hazard"], data["resource"]]
        scored = _margin(z, norm, joint)

        # TELEMETRY ONLY -- recorded so the scored target's choice is transparent and a reader
        # can see it is not the most favourable one. NOT part of any criterion.
        telemetry = {
            "hazard_at_agent": _margin(z, norm, data["hazard"][:, None]),
            "resource_at_agent": _margin(z, norm, data["resource"][:, None]),
            "per_tick_harm_exposure": _margin(z, norm, data["harm_exposure"][:, None]),
        }

        row = {
            "seed": int(seed),
            "arm": "vector_vs_norm",
            "p0h_readiness_met": bool(p0h.get("p0h_readiness_met")),
            "p0h_holdout_lift": (p0h.get("p0h_holdout_vs_constant") or {}).get("lift"),
            "p0h_encoder_tensors_changed": p0h.get("p0h_encoder_tensors_changed"),
            "p0h_precision_norm_applied": p0h.get("p0h_precision_norm_applied"),
            "p0h_target": p0h.get("p0h_target"),
            "n_eval_ticks": int(len(z)),
            "scored_target": SCORED_TARGET,
            "vector_r2": scored["vector_r2"],
            "norm_r2": scored["norm_r2"],
            "margin": scored["margin"],
            "telemetry_other_targets": telemetry,
        }
        cell.stamp(row)

    passed_floor = bool(np.isfinite(row["margin"]) and row["margin"] > MARGIN_ABSOLUTE_FLOOR)
    print("verdict: %s" % ("PASS" if passed_floor else "FAIL"), flush=True)
    return row


def _finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def main(dry_run: bool = False) -> Dict[str, Any]:
    started_at = time.perf_counter()
    seeds = list(SEEDS[:1]) if dry_run else list(SEEDS)
    zg = ZGoalStreamAccumulator()

    arm_results = [_run_seed(s, dry_run, zg) for s in seeds]

    margins = [r["margin"] for r in arm_results if _finite(r["margin"])]
    lifts = [r["p0h_holdout_lift"] for r in arm_results if _finite(r["p0h_holdout_lift"])]
    n_ready = sum(1 for r in arm_results if r["p0h_readiness_met"])
    mean_margin = float(np.mean(margins)) if margins else float("nan")
    sd_margin = float(np.std(margins, ddof=0)) if len(margins) > 1 else 0.0
    noise_band = MARGIN_NOISE_BAND_K * sd_margin

    # --- readiness preconditions: numeric + direction-tagged, so the indexer recomputes `met`
    # from the reported triple rather than trusting the author's boolean. ---
    #
    # DV HEADROOM is the third one and it is not bookkeeping: the margin is
    # (vector R^2 - norm R^2) and vector R^2 <= 1, so the ACHIEVABLE margin is capped at
    # (1 - norm R^2). If the norm already decodes the joint target well, C2's 0.10 absolute
    # floor is out of reach BY CONSTRUCTION and a FAIL would mean nothing. Measuring the cap
    # and self-routing substrate_not_ready is the fix; lowering the pre-registered threshold
    # would not be.
    headroom = [1.0 - r["norm_r2"] for r in arm_results if _finite(r["norm_r2"])]
    min_headroom = float(min(headroom)) if headroom else float("nan")
    preconditions = [
        {"name": "p0h_readiness_met_every_seed",
         "description": "the P0h stage reported a trained encoder on every scored seed",
         "control": "p0h_readiness_met = encoder tensors moved AND held-out aux readout beats "
                    "a constant-mean predictor",
         "measured": float(n_ready), "threshold": float(len(arm_results)),
         "direction": "lower", "comparator": ">=",
         "met": n_ready == len(arm_results)},
        {"name": "p0h_holdout_lift_min_positive",
         "description": "worst-seed P0h held-out lift over a constant-mean predictor",
         "control": "worst cell across seeds, not the mean",
         "measured": (float(min(lifts)) if lifts else float("nan")),
         "threshold": 0.0, "direction": "lower", "comparator": ">",
         "met": bool(lifts) and float(min(lifts)) > 0.0},
        {"name": "dv_headroom_margin_ceiling",
         "description": "worst-seed achievable margin ceiling (1 - norm R^2) must exceed C2's "
                        "absolute floor, or C2 cannot clear by construction",
         "control": "worst cell across seeds, not the mean",
         "measured": min_headroom, "threshold": float(MARGIN_ABSOLUTE_FLOOR),
         "direction": "lower", "comparator": ">",
         "met": bool(_finite(min_headroom) and min_headroom > MARGIN_ABSOLUTE_FLOOR)},
        {"name": "scored_seeds_at_or_above_min",
         "description": "enough seeds produced a finite margin to apply the cross-seed rule",
         "control": "pre-registered MIN_SEEDS",
         "measured": float(len(margins)), "threshold": float(MIN_SEEDS),
         "direction": "lower", "comparator": ">=",
         "met": len(margins) >= MIN_SEEDS},
    ]

    def _precondition(name: str) -> Dict[str, Any]:
        """By NAME, never by index -- a later inserted check must not silently re-point this."""
        for entry in preconditions:
            if entry["name"] == name:
                return entry
        raise KeyError("no precondition named %r" % (name,))

    if dry_run:
        # A smoke runs one seed by design; the seed-count precondition is not a finding here.
        _seed_count = _precondition("scored_seeds_at_or_above_min")
        _seed_count["met"] = True
        _seed_count["control"] += " (relaxed under --dry-run: smoke runs 1 seed)"

    ready = all(p["met"] for p in preconditions)

    # --- criteria (scored ONLY when the preconditions hold) ---
    c1 = bool(ready and _finite(mean_margin) and mean_margin > noise_band)
    c2 = bool(ready and _finite(mean_margin) and mean_margin > MARGIN_ABSOLUTE_FLOOR)
    overall = bool(c1 and c2)

    # C1 is degenerate when the cross-seed SD is exactly 0 -- the band collapses to 0 and C1
    # reduces to "margin > 0", which is NOT the pre-registered test.
    c1_non_degenerate = bool(ready and len(margins) > 1 and sd_margin > 0.0)
    c2_non_degenerate = bool(ready and _finite(mean_margin))

    if not ready:
        label = "substrate_not_ready_requeue"
        routing = ("NOT SCORED -- a readiness precondition failed. Either the encoder was not "
                   "trained on every seed (re-queue at an adequate P0h), or the achievable "
                   "margin ceiling (1 - norm R^2) is below C2's floor, in which case the "
                   "contrast cannot discriminate on this substrate and needs re-specification "
                   "rather than a re-run. Read `interpretation.preconditions` to see which.")
        outcome = "FAIL"
        direction = "non_contributory"
    elif overall:
        label = "vector_beats_norm"
        routing = ("supports SD-086's premise: the 16-d z_harm_a carries decodable structure its "
                   "norm does not, which justifies the trained-head build -- BOUNDED by the "
                   "rank-2 ceiling on harm_obs_a")
        outcome = "PASS"
        direction = "supports"
    else:
        label = "vector_not_distinguishable_from_norm"
        routing = ("weakens SD-086's premise on this substrate and correctly avoids the "
                   "trained-head build; read under the rank-2 ceiling, which bounds BOTH readouts")
        outcome = "FAIL"
        direction = "weakens"

    manifest: Dict[str, Any] = {
        "run_id": "%s_%sZ_v3" % (EXPERIMENT_TYPE,
                                 __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%S")),
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": direction,
        "queue_id": "V3-EXQ-1064",
        "scored_target": SCORED_TARGET,
        "scored_target_rationale": (
            "harm_obs_a is structurally rank 2 (hazard_at_agent in dims [:n], resource_at_agent "
            "in dims [n:]), so the joint pair is exactly the two d.o.f. the substrate has. A "
            "single-scalar target would measure inside the norm's reach."),
        "rank2_limit": (
            "z_harm_a carries at most 2 independent d.o.f. about the world whatever it is trained "
            "on (causal_grid_world.py ~3033-3038); training changes WHICH two, not how many. A "
            "WEAK vector-over-norm margin must NOT be over-read -- the ceiling is low for BOTH "
            "readouts. See REE_assembly/evidence/planning/"
            "harm_obs_a_rank2_scoping_staged_20260918.md"),
        "pre_registered_thresholds": {
            "margin_noise_band_k": MARGIN_NOISE_BAND_K,
            "margin_absolute_floor": MARGIN_ABSOLUTE_FLOOR,
            "min_seeds": MIN_SEEDS,
        },
        "combination_rule": "PASS iff C1 AND C2, with every readiness precondition met",
        "criteria": [
            {"name": "C1_margin_beyond_noise_band", "load_bearing": True, "passed": c1,
             "measured": mean_margin, "threshold": noise_band,
             "detail": "mean margin > %.1f x cross-seed SD (SD=%.6f)"
                       % (MARGIN_NOISE_BAND_K, sd_margin)},
            {"name": "C2_margin_above_absolute_floor", "load_bearing": True, "passed": c2,
             "measured": mean_margin, "threshold": MARGIN_ABSOLUTE_FLOOR},
        ],
        "criteria_non_degenerate": {"C1_margin_beyond_noise_band": c1_non_degenerate,
                                    "C2_margin_above_absolute_floor": c2_non_degenerate},
        "non_degenerate": bool(c1_non_degenerate and c2_non_degenerate),
        "interpretation": {
            "label": label,
            "summary": ("mean vector-minus-norm R^2 margin %.4f (cross-seed SD %.4f, band %.4f) "
                        "on the joint hazard+resource target over %d seed(s)"
                        % (mean_margin, sd_margin, noise_band, len(margins))),
            "routing": routing,
            "preconditions": preconditions,
            "criteria_non_degenerate": {"C1_margin_beyond_noise_band": c1_non_degenerate,
                                        "C2_margin_above_absolute_floor": c2_non_degenerate},
        },
        "arm_results": arm_results,
        "per_seed_margin": {str(r["seed"]): r["margin"] for r in arm_results},
        "per_seed_vector_r2": {str(r["seed"]): r["vector_r2"] for r in arm_results},
        "per_seed_norm_r2": {str(r["seed"]): r["norm_r2"] for r in arm_results},
        "per_seed_p0h_readiness_met": {str(r["seed"]): r["p0h_readiness_met"]
                                       for r in arm_results},
        "custom_information": {
            "telemetry_other_targets_mean_margin": {
                name: float(np.mean([r["telemetry_other_targets"][name]["margin"]
                                     for r in arm_results
                                     if _finite(r["telemetry_other_targets"][name]["margin"])]))
                for name in ("hazard_at_agent", "resource_at_agent", "per_tick_harm_exposure")
            },
            "telemetry_note": ("recorded so the scored target's choice is transparent; NOT part "
                               "of any criterion"),
            "ethics_preflight": {
                "involves_negative_valence": False,
                "involves_suffering_like_state": False,
                "involves_self_model": False,
                "involves_inescapability_or_helplessness": False,
                "involves_offline_replay_over_harm": False,
                "involves_social_mind_or_language": False,
                "involves_human_data_or_clinical_context": False,
                "decision": "allow",
                "note": ("read-only decode of the affective harm channel under SENT-0; V3 is "
                         "pre-ethical instrumentation, not claimed sentient"),
            },
        },
    }
    # Flat scalar readout -- the machine-readable projection the indexer scores on.
    readout: Dict[str, float] = {
        "mean_margin": mean_margin,
        "sd_margin": sd_margin,
        "noise_band": noise_band,
        "mean_vector_r2": float(np.mean([r["vector_r2"] for r in arm_results
                                         if _finite(r["vector_r2"])] or [float("nan")])),
        "mean_norm_r2": float(np.mean([r["norm_r2"] for r in arm_results
                                       if _finite(r["norm_r2"])] or [float("nan")])),
        "min_dv_headroom": min_headroom,
        "n_seeds_scored": float(len(margins)),
        "n_seeds_ready": float(n_ready),
        "c1_margin_beyond_noise_band": 1 if c1 else 0,   # int, not bool -- bools are not numeric
        "c2_margin_above_absolute_floor": 1 if c2 else 0,
        "preconditions_met": 1 if ready else 0,
    }
    manifest["readout"] = {k: v for k, v in readout.items() if _finite(v)}

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=_config_slice(dry_run),
        seeds=seeds,
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=zg.stats(),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-1064 SD-086 option C: z_harm_a vector-vs-norm decode")
    parser.add_argument("--dry-run", action="store_true",
                        help="1 seed, tiny budgets; manifest relocated out of evidence/")
    args = parser.parse_args()

    result = main(dry_run=args.dry_run)
    out_path = result["_out_path"]

    print()
    print("=== V3-EXQ-1064 SD-086 option C: z_harm_a vector vs norm ===")
    print("label:   %s" % result["interpretation"]["label"])
    print("outcome: %s" % result["outcome"])
    print("summary: %s" % result["interpretation"]["summary"])
    print("routing: %s" % result["interpretation"]["routing"])
    print("--- per seed (scored target: %s) ---" % result["scored_target"])
    for r in result["arm_results"]:
        print("  seed=%d ready=%s vector_r2=%.4f norm_r2=%.4f margin=%+.4f"
              % (r["seed"], r["p0h_readiness_met"], r["vector_r2"], r["norm_r2"], r["margin"]))
    print("--- telemetry (NOT scored) ---")
    for name, val in result["custom_information"]["telemetry_other_targets_mean_margin"].items():
        print("  %-26s mean margin %+.4f" % (name, val))
    print("manifest: %s" % out_path)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
