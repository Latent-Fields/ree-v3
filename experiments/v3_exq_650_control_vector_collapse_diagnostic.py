"""V3-EXQ-650: ControlVector logging readiness + Stage-B C_time<->G_vigor collapse.

Substrate-readiness diagnostic for the ControlVector logging telemetry landed
2026-06-07 (recommendation B, four-signal control adjudication). Two purposes:

  1. Confirm the read-only telemetry populates the four control signals
     (V_outcome / C_effort / C_time / G_vigor) and is bit-identical OFF
     (empty _last_control_vector).
  2. Measure the Stage-B collapse EMPIRICALLY: opportunity-cost-of-time
     (C_time) and vigor (G_vigor) are NOT independent -- both are w * v_t for
     the SAME MECH-320 v_t scalar (ARC-068 absorbed into MECH-320; ARC-068
     itself is registered but unbuilt). This converts the collapse from a
     code-reading inference into a measured fact -- the evidence governance
     needs before deciding whether ARC-068 deserves its own scalar (deferred
     post-green-board per the adjudication).

DESIGN: pure telemetry probe, NO training. 3 conditions x 2 seeds.
  ARM_OFF          use_control_vector_logging=False   -> empty telemetry (C3)
  ARM_ON_FLOORED   logging+vigor ON, tonic_vigor_v_t_floor=0.5
                   -> v_t > 0 every tick; the collapse identity is exercised
                      on a NON-DEGENERATE scalar (C1 + load-bearing C2).
  ARM_ON_NATURAL   logging+vigor ON, tonic_vigor_v_t_floor=0.0
                   -> v_t follows the realised EWMA (often gated to 0 -- the
                      documented EXQ-624a sign issue); used to report the
                      across-tick correlation when v_t varies, flagged
                      degenerate otherwise. Informational only.

WHY THE IDENTITY, NOT A CORRELATION, IS LOAD-BEARING: with v_t constant
(the floored arm) a Pearson correlation between C_time and G_vigor is 0/0
(zero variance) -- it would be a vacuous PASS. The robust, always-well-defined
collapse measurement is the per-tick IDENTITY: for every tick with v_t > 0,
C_time_potential == w_passive*v_t AND G_vigor_potential == w_action*v_t. That
holds because both are deterministic functions of one scalar -- which IS the
collapse. The correlation (exactly 1.0 whenever var(v_t) > 0, by collinearity)
is reported as secondary corroboration.

ACCEPTANCE (substrate-readiness, claim_ids=[] -- does NOT weight claim confidence):
  READINESS (load-bearing non-vacuity): ARM_ON_FLOORED n_ticks_vt_positive
    >= VT_POS_FLOOR on >= MIN_SEEDS seeds. Below floor (telemetry never saw a
    non-zero v_t -> the identity is never exercised) self-routes
    substrate_not_ready_requeue, NEVER a substrate verdict.
  C1 four-signal logging: ARM_ON_FLOORED logs V_outcome + C_time + G_vigor with
    present=True on >= MIN_SEEDS seeds (C_effort present is informational --
    requires the dACC to fire, which needs the affective harm stream).
  C2 collapse identity (LOAD-BEARING): ARM_ON_FLOORED per-tick identity max
    deviation < IDENTITY_TOL over all v_t>0 ticks on >= MIN_SEEDS seeds.
  C3 bit-identical OFF: ARM_OFF _last_control_vector empty on every tick, all seeds.
  PASS = readiness met AND C1 AND C2 AND C3.

This experiment trains nothing and writes no memory -- MECH-094 N/A.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

# Repo-root import path.
_THIS = Path(__file__).resolve()
_REE_V3 = _THIS.parent.parent
if str(_REE_V3) not in sys.path:
    sys.path.insert(0, str(_REE_V3))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._harness import StepHarness  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    reset_all_rng,
    compute_arm_fingerprint,
)
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_650_control_vector_collapse_diagnostic"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS: List[str] = []  # substrate-readiness telemetry; weights no claim.

# Pre-registered thresholds (constants, not derived post-hoc).
SEEDS = [42, 43]
MIN_SEEDS = 2
N_EPISODES = 2
STEPS_PER_EPISODE = 200
EPISODES_PER_RUN = N_EPISODES  # denominator for [train] progress prints
VT_POS_FLOOR = 10  # readiness: >= this many v_t>0 ticks in the floored arm
IDENTITY_TOL = 1e-9  # collapse identity max deviation
V_T_FLOOR_VALUE = 0.5  # forced floor in ARM_ON_FLOORED

ARMS = ["ARM_OFF", "ARM_ON_FLOORED", "ARM_ON_NATURAL"]


def _mk_env() -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3)


def _build_config(arm: str, env: CausalGridWorldV2) -> REEConfig:
    base = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    if arm == "ARM_OFF":
        return REEConfig.from_dims(**base)
    cfg = REEConfig.from_dims(
        use_control_vector_logging=True,
        use_tonic_vigor=True,
        use_dacc=True,
        **base,
    )
    cfg.tonic_vigor_v_t_floor = V_T_FLOOR_VALUE if arm == "ARM_ON_FLOORED" else 0.0
    return cfg


def _pearson(xs: List[float], ys: List[float]) -> Any:
    """Pearson correlation; None when either series has ~zero variance."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-18 or syy <= 1e-18:
        return None  # degenerate (constant series) -- collapse shown via identity
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / ((sxx ** 0.5) * (syy ** 0.5))


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    reset_all_rng(seed)
    env = _mk_env()
    cfg = _build_config(arm, env)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, env, train_mode=True, seed=seed)

    n_ticks = 0
    n_off_nonempty = 0  # ARM_OFF violation counter
    n_vt_pos = 0
    present_v = present_e = present_ct = present_gv = 0
    identity_max_dev = 0.0
    ctime_series: List[float] = []
    gvigor_series: List[float] = []
    vt_series: List[float] = []

    for ep in range(N_EPISODES):
        env_reset = env.reset()
        flat_obs, obs_dict = env_reset
        agent.reset()
        harness.reset()
        for _step in range(STEPS_PER_EPISODE):
            r = harness.step(obs_dict)
            n_ticks += 1
            cv = agent._last_control_vector
            if arm == "ARM_OFF":
                if cv:
                    n_off_nonempty += 1
            else:
                if cv:
                    vo = cv.get("V_outcome", {})
                    ce = cv.get("C_effort", {})
                    ct = cv.get("C_time", {})
                    gv = cv.get("G_vigor", {})
                    sh = cv.get("shared", {})
                    present_v += int(bool(vo.get("present")))
                    present_e += int(bool(ce.get("present")))
                    present_ct += int(bool(ct.get("present")))
                    present_gv += int(bool(gv.get("present")))
                    vt = float(sh.get("tonic_vigor_v_t", 0.0))
                    wa = float(sh.get("w_action", 0.0))
                    wp = float(sh.get("w_passive", 0.0))
                    if vt > 0.0:
                        n_vt_pos += 1
                        dev_ct = abs(float(ct.get("potential", 0.0)) - wp * vt)
                        dev_gv = abs(float(gv.get("potential", 0.0)) - wa * vt)
                        identity_max_dev = max(identity_max_dev, dev_ct, dev_gv)
                        ctime_series.append(float(ct.get("potential", 0.0)))
                        gvigor_series.append(float(gv.get("potential", 0.0)))
                        vt_series.append(vt)
            obs_dict = r.next_obs_dict
            if r.done:
                break
        print(
            f"  [train] {arm} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN} "
            f"ticks={n_ticks} vt_pos={n_vt_pos}",
            flush=True,
        )

    corr = _pearson(ctime_series, gvigor_series)
    vt_var = 0.0
    if len(vt_series) >= 2:
        mvt = sum(vt_series) / len(vt_series)
        vt_var = sum((v - mvt) ** 2 for v in vt_series) / len(vt_series)

    row = {
        "arm": arm,
        "seed": seed,
        "n_ticks": n_ticks,
        "n_off_nonempty": n_off_nonempty,
        "n_vt_positive": n_vt_pos,
        "present_V_outcome": present_v,
        "present_C_effort": present_e,
        "present_C_time": present_ct,
        "present_G_vigor": present_gv,
        "identity_max_dev": identity_max_dev,
        "ctime_gvigor_corr": corr,
        "vt_variance": vt_var,
    }
    row["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice=cfg.__dict__ if hasattr(cfg, "__dict__") else {},
        seed=seed,
        script_path=_THIS,
        rng_fully_reset=True,
    )
    return row


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = SEEDS[:1] if dry_run else SEEDS
    global N_EPISODES, STEPS_PER_EPISODE
    if dry_run:
        N_EPISODES = 1
        STEPS_PER_EPISODE = 12

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            row = _run_cell(arm, seed)
            rows.append(row)
            print("verdict: PASS", flush=True)  # per-cell completion marker

    def _by(arm: str) -> List[Dict[str, Any]]:
        return [r for r in rows if r["arm"] == arm]

    # READINESS: floored arm must have exercised the identity on v_t>0 ticks.
    floored = _by("ARM_ON_FLOORED")
    seeds_vt_ok = sum(1 for r in floored if r["n_vt_positive"] >= VT_POS_FLOOR)
    readiness_met = seeds_vt_ok >= min(MIN_SEEDS, len(seeds))

    # C1: four-signal logging (V_outcome + C_time + G_vigor present) in floored arm.
    seeds_c1 = sum(
        1
        for r in floored
        if r["present_V_outcome"] > 0
        and r["present_C_time"] > 0
        and r["present_G_vigor"] > 0
    )
    c1 = seeds_c1 >= min(MIN_SEEDS, len(seeds))

    # C2 (LOAD-BEARING): per-tick collapse identity holds on v_t>0 ticks.
    seeds_c2 = sum(
        1
        for r in floored
        if r["n_vt_positive"] >= VT_POS_FLOOR and r["identity_max_dev"] < IDENTITY_TOL
    )
    c2 = seeds_c2 >= min(MIN_SEEDS, len(seeds))

    # C3: bit-identical OFF (no telemetry written).
    off = _by("ARM_OFF")
    c3 = all(r["n_off_nonempty"] == 0 for r in off) and len(off) > 0

    # Secondary: correlation across the natural arm (exact 1.0 when v_t varies).
    natural = _by("ARM_ON_NATURAL")
    nat_corrs = [r["ctime_gvigor_corr"] for r in natural if r["ctime_gvigor_corr"] is not None]
    nat_corr_mean = sum(nat_corrs) / len(nat_corrs) if nat_corrs else None
    nat_degenerate = len(nat_corrs) == 0

    passed = bool(readiness_met and c1 and c2 and c3)
    outcome = "PASS" if passed else "FAIL"

    if not readiness_met:
        label = "substrate_not_ready_requeue"
    elif passed:
        label = "control_vector_logging_ready_collapse_confirmed"
    else:
        label = "control_vector_logging_criteria_unmet"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "vt_positive_ticks_floored_arm",
                "description": (
                    "ARM_ON_FLOORED logged >= VT_POS_FLOOR ticks with v_t>0 so "
                    "the collapse identity is exercised on a non-degenerate scalar"
                ),
                "measured": max((r["n_vt_positive"] for r in floored), default=0),
                "threshold": VT_POS_FLOOR,
                "control": "tonic_vigor_v_t_floor=0.5 forces v_t>0 every tick",
                "met": readiness_met,
            }
        ],
        "criteria": [
            {"name": "C2_collapse_identity", "load_bearing": True, "passed": c2},
        ],
        "criteria_non_degenerate": {
            # C1 non-degenerate: signals actually present (not vacuously absent).
            "C1": bool(seeds_c1 > 0),
            # C2 non-degenerate: identity measured over real v_t>0 ticks.
            "C2": bool(seeds_c2 > 0 and all(r["n_vt_positive"] > 0 for r in floored)),
            # C3 non-degenerate: OFF arm actually stepped (n_ticks>0).
            "C3": bool(all(r["n_ticks"] > 0 for r in off) and len(off) > 0),
        },
        "notes": (
            "Collapse is proven by the per-tick identity (C_time=w_passive*v_t, "
            "G_vigor=w_action*v_t for ONE v_t), robust to constant v_t. The "
            "across-tick correlation is exactly 1.0 when v_t varies (collinear) "
            "and reported degenerate when v_t is constant."
        ),
    }

    run_id = (
        f"{EXPERIMENT_TYPE}_"
        f"{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "acceptance": {
            "readiness_met": readiness_met,
            "C1_four_signal_logging": c1,
            "C2_collapse_identity_load_bearing": c2,
            "C3_bit_identical_off": c3,
            "seeds_vt_ok": seeds_vt_ok,
            "seeds_c1": seeds_c1,
            "seeds_c2": seeds_c2,
            "VT_POS_FLOOR": VT_POS_FLOOR,
            "IDENTITY_TOL": IDENTITY_TOL,
        },
        "secondary": {
            "natural_arm_corr_mean": nat_corr_mean,
            "natural_arm_corr_degenerate": nat_degenerate,
            "natural_arm_note": (
                "corr is exactly 1.0 when v_t varies; degenerate (None) when "
                "v_t pinned (EXQ-624a) -- collapse still proven via C2 identity"
            ),
        },
        "interpretation": interpretation,
        "rows": rows,
        "dry_run": dry_run,
    }
    return manifest


def _write_manifest(manifest: Dict[str, Any]) -> Path:
    out_dir = (_REE_V3.parent / "REE_assembly" / "evidence" / "experiments")
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_experiment(dry_run=args.dry_run)
    out_path = _write_manifest(result)
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"manifest: {out_path}", flush=True)

    _oc = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_oc if _oc in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
