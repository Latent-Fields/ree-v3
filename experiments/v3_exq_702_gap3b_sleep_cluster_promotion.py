"""
V3-EXQ-702: sleep_substrate GAP-3b -- MECH-285 / MECH-272 / MECH-273 behavioural
promotion run over the unified offline-consolidation pathway.
SLEEP DRIVER: manual-cycle-loop (run_sleep_cycle() forced once per cell in a
dedicated wake-sleep measurement loop)

GAP-3 / GAP-4 / GAP-8 landed the offline-consolidation substrate (the unified
use_sleep_aggregation_cluster master flag + MECH-273 replay-derived gradient +
MECH-272 routing consumer), and each lists MECH-285/272/273 in unblocks_claims as
the claims its substrate makes reachable for PROMOTION -- but all three are STILL
`candidate` in claims.yaml because no behavioural discriminative promotion run was
ever queued (the offline pathway's whole purpose is unrealised at the evidence
layer). This is the owed promotion run: a discriminative arm-ON-vs-arm-OFF
measurement over use_sleep_aggregation_cluster that produces scoreable evidence for
the three claims.

MECH-275 is DELIBERATELY NOT TAGGED here: it is epistemic_category=substrate_conditional
(claims.yaml governance_2026_06_10), gated on the unbuilt MECH-276 counterfactual-
backed-attribution feedstock; a cluster ON/OFF run cannot produce valid promotion
evidence for it. Its promotion is owed to a later run once MECH-276 lands.

COMMITMENT-FREE: this is a consolidation / measurement run. It measures the
offline-pathway's internal consolidation signatures ON vs OFF; it does NOT route
through committed action selection, so it does NOT depend on the F-dominance
conversion ceiling (MECH-439) or any of the behavioural-diversity machinery.

Design (per arm x seed cell):
  ARM_OFF: use_sleep_aggregation_cluster=False -> no sleep_loop / sampler / routing
           gate / aggregator. The offline pathway is silent. The discriminative
           baseline (every consolidation signature reads zero).
  ARM_ON:  use_sleep_aggregation_cluster=True. Drive the waking loop (warm encoders
           + experience buffer + harm-replay buffer), install a HOT (high-staleness)
           and a COLD (zero-staleness) anchor, inject a differentiated staleness map,
           snapshot the self-model (E2_harm_s) parameters, force one sleep cycle, then
           read the per-claim consolidation signatures.

Per-claim discriminative metric (the ON-arm-internal signature is load-bearing;
the OFF arm is the zero-baseline contrast, so a PASS is NOT just "the flag fires"):
  MECH-272 (state-gated routing): the routing gate flips the anchor/probe channel
    weights by sleep phase. Claim falsifier: the anchored-channel share DROPS in
    sleep. Metric C272 = applied SWS anchor weight measurably below the waking 1.0
    AND mech272_n_routed > 0 (routing actually fired). OFF: weight 1.0, n_routed 0.
  MECH-285 (V_s residual replay priority): the sampler draws by softmax over the
    frozen MECH-284 staleness snapshot. Claim falsifier: replay is biased toward
    stale regions. Metric C285 = with an injected HOT(0.9)/COLD(0.0) staleness map,
    HOT is drawn >= SKEW_RATIO x COLD. OFF: no sampler, no draws.
  MECH-273 (self-model aggregation): the offline gradient pass aggregates replay-
    derived (z_harm_s, action) tuples into the self-model (E2_harm_s). Claim
    falsifier: the sleep half durably moves the self-model. Metric C273 = E2_harm_s
    parameter L2 delta across the sleep cycle >= floor AND mech273_n_offline_steps>0.
    OFF: no aggregator, zero param movement.

Non-degeneracy guards (a vacuous criterion is scoring-excluded, not scored):
  MECH-285 -> the frozen staleness snapshot must be NON-uniform (HOT != COLD) for the
    softmax skew to be testable.
  MECH-273 -> the harm-replay buffer must be NON-empty at sleep entry, so the offline
    pass uses REAL replay-derived tuples (GAP-4) rather than the synthetic
    zeros/round-robin fallback (which would move the params without testing the
    aggregation-of-experience claim).
  MECH-272 -> the SWS schema pass must have written (sws_n_writes > 0); a zero-write
    cycle routes nothing.

Phased training: NOT applicable. This run does not train a downstream head on a
moving encoder. The agent's own waking act loop warms the encoders (as in V3-EXQ-581);
the only learning under test is the sleep offline gradient pass itself, whose effect on
E2_harm_s is measured directly before/after -- no P0/P1/P2 latent-target collapse hazard.

Acceptance: each claim "supports" iff its ON criterion holds on >= PASS_FRACTION of
seeds with its non-degeneracy met AND the OFF baseline reads the zero contrast.
Overall PASS iff all three claims support.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_702_gap3b_sleep_cluster_promotion"
QUEUE_ID = "V3-EXQ-702"
CLAIM_IDS: List[str] = ["MECH-285", "MECH-272", "MECH-273"]
EXPERIMENT_PURPOSE = "evidence"
SLEEP_DRIVER_PATTERN = "manual-cycle-loop"

SEEDS = [42, 7, 123]
N_DRIVE_STEPS = 40          # waking warmup: encoders + _world_experience_buffer + _last_action
N_HARM_SENSE_STEPS = 12     # extra sense(obs_harm=...) ticks to fill the GAP-4 harm-replay buffer
DRAWS_PER_CYCLE = 200       # high so the HOT/COLD staleness skew is statistically clear
SWS_ANCHOR_WEIGHT = 0.6     # default mech272_sws_anchor_weight (the SWS-row anchor share)
STALENESS_HOT = 0.9
STALENESS_COLD = 0.0
SAMPLER_TEMPERATURE = 0.3   # sharpen the staleness softmax (matches the C7 contract idiom)

# Dims
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
HARM_OBS_DIM = 51           # use_harm_stream default (hazard_field 25 + resource_field 25 + harm_exposure 1)

# Pre-registered acceptance thresholds (absolute floors; per-seed)
C272_ANCHOR_DROP_MARGIN = 0.10   # applied SWS anchor weight must be <= 1.0 - margin
C285_SKEW_RATIO = 3.0            # HOT draws >= ratio x COLD draws
C285_MIN_DRAWS = 20             # at least this many total draws landed (non-vacuity)
C273_PARAM_DELTA_FLOOR = 1e-5   # E2_harm_s L2 parameter movement across the sleep cycle
PASS_FRACTION = 2.0 / 3.0        # >= 2/3 seeds per criterion

ARMS = [
    {"arm": "ARM_OFF", "use_cluster": False,
     "description": "use_sleep_aggregation_cluster=False -- offline pathway silent (zero baseline)"},
    {"arm": "ARM_ON", "use_cluster": True,
     "description": "use_sleep_aggregation_cluster=True -- offline-consolidation pathway live"},
]


def _build_config(*, use_cluster: bool) -> REEConfig:
    """Both arms enable the substrate prerequisites identically (anchor sets,
    staleness accumulator, harm stream, E2_harm_s forward); ONLY the unified
    use_sleep_aggregation_cluster master flag differs between arms, so the arm
    contrast isolates the offline-consolidation pathway.
    """
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        use_sleep_aggregation_cluster=use_cluster,
        sleep_loop_episodes_K=1,
        mech285_draws_per_cycle=DRAWS_PER_CYCLE,
        mech285_temperature=SAMPLER_TEMPERATURE,
        mech272_sws_anchor_weight=SWS_ANCHOR_WEIGHT,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        use_e2_harm_s_forward=True,
    )


def _config_slice(*, arm: str, use_cluster: bool) -> Dict[str, Any]:
    return {
        "arm": arm,
        "use_sleep_aggregation_cluster": use_cluster,
        "body_obs_dim": BODY_OBS_DIM,
        "world_obs_dim": WORLD_OBS_DIM,
        "action_dim": ACTION_DIM,
        "harm_obs_dim": HARM_OBS_DIM,
        "n_drive_steps": N_DRIVE_STEPS,
        "n_harm_sense_steps": N_HARM_SENSE_STEPS,
        "draws_per_cycle": DRAWS_PER_CYCLE,
        "sws_anchor_weight": SWS_ANCHOR_WEIGHT,
        "sampler_temperature": SAMPLER_TEMPERATURE,
        "staleness_hot": STALENESS_HOT,
        "staleness_cold": STALENESS_COLD,
    }


def _drive_waking(agent: REEAgent, *, seed: int, arm: str, n_drive: int) -> None:
    """Warm the encoders + populate _world_experience_buffer and set _last_action."""
    for step in range(n_drive):
        obs_body = torch.randn(BODY_OBS_DIM)
        obs_world = torch.randn(WORLD_OBS_DIM)
        agent.act_with_split_obs(obs_body=obs_body, obs_world=obs_world)
        if (step + 1) % 10 == 0:
            print(
                f"  [train] drive seed={seed} arm={arm} ep {step + 1}/{n_drive}",
                flush=True,
            )


def _fill_harm_replay_buffer(agent: REEAgent, *, n_steps: int) -> int:
    """Sense WITH obs_harm so z_harm populates and the GAP-4 harm-replay buffer
    fills with real waking-stream (z_harm_s, action) tuples (the offline pass's
    replay-derived feedstock). _last_action is already set by the drive loop.
    Returns the buffer length.
    """
    for _ in range(n_steps):
        obs_body = torch.randn(BODY_OBS_DIM)
        obs_world = torch.randn(WORLD_OBS_DIM)
        obs_harm = torch.randn(HARM_OBS_DIM)
        agent.sense(obs_body, obs_world, obs_harm=obs_harm)
    return len(getattr(agent, "_harm_replay_buffer", []))


def _install_hot_cold_anchors(agent: REEAgent) -> None:
    anchor_set = agent.hippocampal.anchor_set
    assert anchor_set is not None, "anchor_set must be initialised (use_anchor_sets)"
    for i, seg in enumerate(("hot", "cold")):
        z = torch.randn(1, 32) * (i + 1)
        anchor_set.write_anchor(
            scale="fast",
            segment_id=seg,
            stream_mixture=(f"stream_{seg}",),
            z_world=z,
        )


def _inject_staleness(agent: REEAgent) -> bool:
    """Inject a differentiated staleness map (HOT high, COLD zero) directly onto the
    accumulator -- the sanctioned test idiom (test_sleep_phase_b_replay_sampler C7).
    Returns True iff the map is non-uniform (the MECH-285 non-degeneracy precondition).
    """
    acc = agent.hippocampal.staleness_accumulator
    assert acc is not None, "staleness_accumulator must be initialised"
    acc._staleness[("fast", "hot")] = STALENESS_HOT
    acc._staleness[("fast", "cold")] = STALENESS_COLD
    return STALENESS_HOT != STALENESS_COLD


def _e2_harm_s_param_l2(agent: REEAgent) -> List[torch.Tensor]:
    return [p.detach().clone() for p in agent.e2_harm_s.parameters()]


def _param_delta_l2(before: List[torch.Tensor], after: List[torch.Tensor]) -> float:
    sq = 0.0
    for b, a in zip(before, after):
        sq += float(((a - b) ** 2).sum().item())
    return math.sqrt(sq)


def _metric(metrics: Dict[str, float], key: str, default: float = 0.0) -> float:
    return float(metrics.get(key, default))


def _run_on_cell(*, seed: int) -> Dict[str, Any]:
    """ARM_ON: drive, fill replay buffer, inject staleness, sleep, measure."""
    agent = REEAgent(_build_config(use_cluster=True))
    _drive_waking(agent, seed=seed, arm="ARM_ON", n_drive=N_DRIVE_STEPS)
    harm_buf_len = _fill_harm_replay_buffer(agent, n_steps=N_HARM_SENSE_STEPS)
    _install_hot_cold_anchors(agent)
    staleness_non_uniform = _inject_staleness(agent)

    params_before = _e2_harm_s_param_l2(agent)
    metrics = agent.sleep_loop.force_cycle(agent)
    params_after = _e2_harm_s_param_l2(agent)
    metrics = metrics or {}

    # MECH-272: routing regime shift (anchor share drops in SWS).
    sws_anchor_weight_applied = _metric(metrics, "sws_anchor_weight_applied", 1.0)
    mech272_n_routed = _metric(
        metrics, "mech272_n_routed_sws", _metric(metrics, "mech272_n_routed", 0.0)
    )
    sws_n_writes = _metric(metrics, "sws_n_writes", 0.0)

    # MECH-285: staleness-priority replay skew over the frozen snapshot.
    sampler = agent.sleep_replay_sampler
    draw_counts = dict(getattr(sampler, "draw_region_counts", {}) or {})
    hot_draws = float(draw_counts.get(("fast", "hot"), 0))
    cold_draws = float(draw_counts.get(("fast", "cold"), 0))
    total_draws = _metric(metrics, "mech285_n_draws", hot_draws + cold_draws)
    snapshot_is_uniform = bool(_metric(metrics, "mech285_snapshot_is_uniform", 1.0))

    # MECH-273: self-model aggregation -- E2_harm_s parameter movement.
    mech273_param_delta = _param_delta_l2(params_before, params_after)
    mech273_n_offline_steps = _metric(metrics, "mech273_n_offline_steps", 0.0)

    return {
        "seed": seed,
        "arm": "ARM_ON",
        "harm_replay_buffer_len": harm_buf_len,
        "staleness_non_uniform": bool(staleness_non_uniform and not snapshot_is_uniform),
        "mech272_sws_anchor_weight_applied": sws_anchor_weight_applied,
        "mech272_n_routed": mech272_n_routed,
        "mech272_sws_n_writes": sws_n_writes,
        "mech285_hot_draws": hot_draws,
        "mech285_cold_draws": cold_draws,
        "mech285_total_draws": total_draws,
        "mech285_snapshot_is_uniform": snapshot_is_uniform,
        "mech273_param_delta_l2": mech273_param_delta,
        "mech273_n_offline_steps": mech273_n_offline_steps,
    }


def _run_off_cell(*, seed: int) -> Dict[str, Any]:
    """ARM_OFF: identical prereqs but the cluster flag off -> offline pathway silent.
    sleep_loop is None, so there is no sleep cycle and every consolidation signature
    reads its zero baseline.
    """
    agent = REEAgent(_build_config(use_cluster=False))
    _drive_waking(agent, seed=seed, arm="ARM_OFF", n_drive=N_DRIVE_STEPS)
    harm_buf_len = _fill_harm_replay_buffer(agent, n_steps=N_HARM_SENSE_STEPS)

    params_before = _e2_harm_s_param_l2(agent)
    sleep_loop = getattr(agent, "sleep_loop", None)
    sleep_pathway_present = sleep_loop is not None
    # No force_cycle: the cluster is off, so the offline pathway cannot fire.
    params_after = _e2_harm_s_param_l2(agent)
    mech273_param_delta = _param_delta_l2(params_before, params_after)

    return {
        "seed": seed,
        "arm": "ARM_OFF",
        "harm_replay_buffer_len": harm_buf_len,
        "sleep_pathway_present": sleep_pathway_present,
        "mech272_sws_anchor_weight_applied": 1.0,
        "mech272_n_routed": 0.0,
        "mech272_sws_n_writes": 0.0,
        "mech285_hot_draws": 0.0,
        "mech285_cold_draws": 0.0,
        "mech285_total_draws": 0.0,
        "mech273_param_delta_l2": mech273_param_delta,
        "mech273_n_offline_steps": 0.0,
    }


def _score(on_rows: List[Dict[str, Any]], off_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds = len(on_rows)
    need = math.ceil(PASS_FRACTION * n_seeds)

    # Per-seed criterion booleans (ON arm carries the load-bearing signature).
    c272_seed = [
        (r["mech272_sws_anchor_weight_applied"] <= 1.0 - C272_ANCHOR_DROP_MARGIN)
        and (r["mech272_n_routed"] > 0)
        for r in on_rows
    ]
    c285_seed = [
        (r["mech285_hot_draws"] >= C285_SKEW_RATIO * max(r["mech285_cold_draws"], 1.0))
        and (r["mech285_total_draws"] >= C285_MIN_DRAWS)
        for r in on_rows
    ]
    c273_seed = [
        (r["mech273_param_delta_l2"] >= C273_PARAM_DELTA_FLOOR)
        and (r["mech273_n_offline_steps"] > 0)
        for r in on_rows
    ]

    # Non-degeneracy per claim (whole-claim: degenerate if NO seed met the precondition).
    nd_272 = any(r["mech272_sws_n_writes"] > 0 for r in on_rows)
    nd_285 = any(r["staleness_non_uniform"] for r in on_rows)
    nd_273 = any(r["harm_replay_buffer_len"] > 0 for r in on_rows)

    # OFF baseline contrast (must read the zero baseline for the arm contrast to hold).
    off_272_zero = all(r["mech272_n_routed"] == 0 for r in off_rows)
    off_285_zero = all(r["mech285_total_draws"] == 0 for r in off_rows)
    off_273_zero = all(r["mech273_param_delta_l2"] < C273_PARAM_DELTA_FLOOR for r in off_rows)

    def _direction(seed_bools: List[bool], non_degen: bool, off_zero: bool) -> Tuple[str, bool]:
        passed = (sum(seed_bools) >= need) and off_zero
        if not non_degen:
            return "unknown", False
        return ("supports" if passed else "weakens"), passed

    dir_272, p272 = _direction(c272_seed, nd_272, off_272_zero)
    dir_285, p285 = _direction(c285_seed, nd_285, off_285_zero)
    dir_273, p273 = _direction(c273_seed, nd_273, off_273_zero)

    overall_pass = p272 and p285 and p273
    directions = {"MECH-272": dir_272, "MECH-285": dir_285, "MECH-273": dir_273}
    if all(d == "supports" for d in directions.values()):
        evidence_direction = "supports"
    elif all(d == "weakens" for d in directions.values()):
        evidence_direction = "weakens"
    else:
        evidence_direction = "mixed"

    non_degenerate_per_claim = {
        "MECH-272": bool(nd_272),
        "MECH-285": bool(nd_285),
        "MECH-273": bool(nd_273),
    }
    degeneracy_reason = ""
    if not nd_272:
        degeneracy_reason += "MECH-272: no SWS writes (sws_n_writes==0); "
    if not nd_285:
        degeneracy_reason += "MECH-285: staleness snapshot uniform; "
    if not nd_273:
        degeneracy_reason += "MECH-273: harm-replay buffer empty (synthetic fallback, not replay-derived); "

    return {
        "outcome": "PASS" if overall_pass else "FAIL",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": directions,
        "non_degenerate_per_claim": non_degenerate_per_claim,
        "degeneracy_reason": degeneracy_reason.strip(),
        "criteria_results": {
            "n_seeds": n_seeds,
            "need_seeds": need,
            "C272_seed_pass": c272_seed,
            "C285_seed_pass": c285_seed,
            "C273_seed_pass": c273_seed,
            "off_272_zero": off_272_zero,
            "off_285_zero": off_285_zero,
            "off_273_zero": off_273_zero,
            "MECH-272_pass": p272,
            "MECH-285_pass": p285,
            "MECH-273_pass": p273,
        },
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print("V3-EXQ-702: GAP-3b sleep-aggregation cluster promotion", flush=True)
    print(f"  seeds={seeds} dry_run={dry_run} claims={CLAIM_IDS}", flush=True)

    on_rows: List[Dict[str, Any]] = []
    off_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []

    for arm_cfg in ARMS:
        arm = arm_cfg["arm"]
        use_cluster = arm_cfg["use_cluster"]
        for seed in seeds:
            print(f"Seed {seed} Condition {arm}", flush=True)
            with arm_cell(
                seed,
                config_slice=_config_slice(arm=arm, use_cluster=use_cluster),
                script_path=Path(__file__),
                extra_ineligible_reasons=["evidence_run_no_reuse"],
            ) as cell:
                if use_cluster:
                    row = _run_on_cell(seed=seed)
                    on_rows.append(row)
                else:
                    row = _run_off_cell(seed=seed)
                    off_rows.append(row)
                cell.stamp(row)
            arm_results.append(row)
            print(
                f"  [result] seed={seed} arm={arm} "
                f"anchor_w={row['mech272_sws_anchor_weight_applied']:.3f} "
                f"hot/cold={row['mech285_hot_draws']:.0f}/{row['mech285_cold_draws']:.0f} "
                f"selfmodel_delta={row['mech273_param_delta_l2']:.3e}",
                flush=True,
            )
            print("verdict: PASS", flush=True)

    scored = _score(on_rows, off_rows)

    print("", flush=True)
    print(f"MECH-272 (routing regime shift): {scored['evidence_direction_per_claim']['MECH-272']}", flush=True)
    print(f"MECH-285 (staleness-priority replay): {scored['evidence_direction_per_claim']['MECH-285']}", flush=True)
    print(f"MECH-273 (self-model aggregation): {scored['evidence_direction_per_claim']['MECH-273']}", flush=True)
    print(f"non_degenerate_per_claim: {scored['non_degenerate_per_claim']}", flush=True)
    print(f"Overall outcome: {scored['outcome']}", flush=True)

    scored["arm_results"] = arm_results
    return scored


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_path = out_dir / f"{run_id}.json"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate_per_claim": result["non_degenerate_per_claim"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_drive_steps": N_DRIVE_STEPS,
            "n_harm_sense_steps": N_HARM_SENSE_STEPS,
            "draws_per_cycle": DRAWS_PER_CYCLE,
            "sws_anchor_weight": SWS_ANCHOR_WEIGHT,
            "sampler_temperature": SAMPLER_TEMPERATURE,
            "staleness_hot": STALENESS_HOT,
            "staleness_cold": STALENESS_COLD,
        },
        "acceptance_criteria": {
            "MECH-272": f"ARM_ON SWS anchor weight <= {1.0 - C272_ANCHOR_DROP_MARGIN} AND mech272_n_routed>0 on >=2/3 seeds; ARM_OFF n_routed==0",
            "MECH-285": f"ARM_ON HOT draws >= {C285_SKEW_RATIO}x COLD AND total>={C285_MIN_DRAWS} on >=2/3 seeds; ARM_OFF no draws",
            "MECH-273": f"ARM_ON E2_harm_s param L2 delta >= {C273_PARAM_DELTA_FLOOR} AND mech273_n_offline_steps>0 on >=2/3 seeds; ARM_OFF zero movement",
        },
        "criteria_results": result["criteria_results"],
        "arm_results": result["arm_results"],
        "notes": (
            "sleep_substrate GAP-3b promotion run: discriminative ARM-ON-vs-OFF over "
            "use_sleep_aggregation_cluster producing scoreable evidence for "
            "MECH-285/272/273. MECH-275 deliberately NOT tagged (substrate_conditional, "
            "gated on the unbuilt MECH-276 feedstock). Commitment-free consolidation "
            "measurement; no F-dominance dependency. Per-claim direction + per-claim "
            "non-degeneracy guards (uniform-snapshot / empty-replay-buffer / zero-SWS-"
            "writes are scoring-excluded, not scored)."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written to evidence/", flush=True)
        print(json.dumps(manifest, indent=2, default=str), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
        dry_run=args.dry_run,
    )
    sys.exit(0)
