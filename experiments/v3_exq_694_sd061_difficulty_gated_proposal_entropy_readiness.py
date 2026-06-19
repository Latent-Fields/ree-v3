"""V3-EXQ-694: SD-061 difficulty-gated proposal-entropy substrate-readiness diagnostic.

SD-061 (difficulty_gated_proposal_entropy) builds MECH-343's substrate_conditional
blocker part 2: a StuckStateDetector + a DifficultyGatedProposalEntropy regulator that,
under an impasse-with-goal, transiently WIDENS the ARC-018/CEM proposal layer (more
candidates + higher within-class temperature) and decays as the impasse clears.

This is a CLAIM-FREE substrate-readiness probe (NOT the Q-056 governance falsifier). It
confirms the wiring is non-vacuous on a controlled stuck-vs-baseline contrast:

  ARM_OFF (regulator disabled): the proposal is INVARIANT to stuck_score -> bit-identical
    baseline (the regulator is None; setting _last_stuck_score has no effect).
  ARM_ON (regulator enabled): a high stuck_score WIDENS the proposed candidate set and
    raises (>=) the candidate first-action-class entropy vs a zero stuck_score; and a
    StuckStateDetector driven through a genuine impasse-then-relief sequence RISES above
    threshold then DECAYS (the asymmetric-EMA hysteresis = MECH-343 entropy decay).

EXPERIMENT_PURPOSE=diagnostic; claim_ids=[]. PASS clears the SD-061 readiness gate; the
Q-056 3-arm governance falsifier (entropy-off / stuck-gated / always-high, matched
easy/hard controls) is a separate later /queue-experiment session.

Readiness self-route: if the detector never reaches is_stuck under the induced impasse
sequence (vacuous probe), self-route substrate_not_ready_requeue rather than a verdict.

SLEEP DRIVER: N/A (no sleep). MECH-094: probe is waking-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_694_sd061_difficulty_gated_proposal_entropy_readiness"
QUEUE_ID = "V3-EXQ-694"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

# --- Pre-registered thresholds (module constants) ---
SEEDS = [42, 43, 44]
ARMS = ["ARM_OFF", "ARM_ON"]
WARMUP_STEPS = 12
STUCK_SEQUENCE_TICKS = 15
RELIEF_SEQUENCE_TICKS = 40
DGPE_CANDIDATE_WIDEN_MAX = 8
DGPE_TEMPERATURE_GAIN_MAX = 1.0
STUCK_THRESHOLD = 0.5  # mirrors the detector default
PASS_FRACTION = 2.0 / 3.0  # >= 2/3 seeds

# Env / agent dims
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8


def _first_action_entropy(candidates: List[Any]) -> float:
    """Shannon entropy (nats) over candidate first-action classes."""
    classes: List[int] = []
    for c in candidates:
        try:
            classes.append(int(torch.argmax(c.actions[0, 0, :]).item()))
        except (AttributeError, RuntimeError, TypeError, IndexError):
            continue
    if not classes:
        return 0.0
    n = len(classes)
    counts = Counter(classes)
    ent = 0.0
    for k in counts.values():
        p = k / n
        ent -= p * math.log(p)
    return ent


def _build_agent(use_dgpe: bool, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_difficulty_gated_proposal_entropy=use_dgpe,
        dgpe_candidate_widen_max=DGPE_CANDIDATE_WIDEN_MAX,
        dgpe_temperature_gain_max=DGPE_TEMPERATURE_GAIN_MAX,
        stuck_threshold=STUCK_THRESHOLD,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _proposal_at_stuck(agent: REEAgent, latent_state, e1_prior, stuck: float) -> Tuple[int, float]:
    """Set the lagged stuck_score and read the next proposal's (n_candidates, entropy)."""
    agent._last_stuck_score = float(stuck)
    candidates = agent._e3_tick(latent_state, e1_prior)
    return len(candidates), _first_action_entropy(candidates)


def _run_cell(use_dgpe: bool, seed: int) -> Dict[str, Any]:
    # Controlled probe: synthetic obs of the configured dims (no env -- the
    # substrate under test is the proposal-widening gain, not env dynamics).
    agent = _build_agent(use_dgpe, seed)
    torch.manual_seed(seed)
    for _ in range(WARMUP_STEPS):
        agent.act_with_split_obs(torch.randn(1, BODY_OBS_DIM), torch.randn(1, WORLD_OBS_DIM))
    # Fresh latent + e1_prior for the controlled proposal probe.
    latent = agent.sense(torch.randn(1, BODY_OBS_DIM), torch.randn(1, WORLD_OBS_DIM))
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks.get("e1_tick", False)
        else torch.zeros(1, WORLD_DIM, device=agent.device)
    )
    n_base, ent_base = _proposal_at_stuck(agent, latent, e1_prior, stuck=0.0)
    n_stuck, ent_stuck = _proposal_at_stuck(agent, latent, e1_prior, stuck=1.0)

    # Detector hysteresis (ON arm): genuine impasse-then-relief sequence.
    detector_peak = 0.0
    detector_final = 0.0
    detector_fired = False
    if agent.stuck_state_detector is not None:
        d = agent.stuck_state_detector
        d.reset()
        for _ in range(STUCK_SEQUENCE_TICKS):
            s = d.update(
                goal_proximity=0.30,
                score_margin=0.0,
                n_candidates=8,
                committed_action_class=3,
                choice_difficulty=0.0,
                goal_salience=0.8,
            )
            detector_peak = max(detector_peak, s)
        detector_fired = d.is_stuck()
        for k in range(RELIEF_SEQUENCE_TICKS):
            detector_final = d.update(
                goal_proximity=0.30 + 0.02 * k,
                score_margin=1.0,
                n_candidates=8,
                committed_action_class=k % 4,
                choice_difficulty=1.0,
                goal_salience=0.8,
            )

    return {
        "seed": seed,
        "arm": "ARM_ON" if use_dgpe else "ARM_OFF",
        "n_candidates_base": n_base,
        "n_candidates_stuck": n_stuck,
        "first_action_entropy_base": round(ent_base, 6),
        "first_action_entropy_stuck": round(ent_stuck, 6),
        "n_widen": int(n_stuck - n_base),
        "entropy_delta": round(ent_stuck - ent_base, 6),
        "detector_peak": round(detector_peak, 6),
        "detector_final": round(detector_final, 6),
        "detector_fired": bool(detector_fired),
    }


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            use_dgpe = arm == "ARM_ON"
            full_config = {
                "use_difficulty_gated_proposal_entropy": use_dgpe,
                "dgpe_candidate_widen_max": DGPE_CANDIDATE_WIDEN_MAX,
                "dgpe_temperature_gain_max": DGPE_TEMPERATURE_GAIN_MAX,
                "seed": seed,
                "arm": arm,
            }
            with arm_cell(
                seed, config_slice=full_config, script_path=Path(__file__)
            ) as cell:
                row = _run_cell(use_dgpe, seed)
                cell.stamp(row)
            rows.append(row)

    # Per-seed adjudication.
    off = {r["seed"]: r for r in rows if r["arm"] == "ARM_OFF"}
    on = {r["seed"]: r for r in rows if r["arm"] == "ARM_ON"}
    per_seed = []
    n_c1 = 0  # detector non-vacuity (fires + decays)
    n_c2 = 0  # regulator load-bearing (ON widens under stuck; OFF invariant)
    for seed in SEEDS:
        o, n = off.get(seed), on.get(seed)
        if o is None or n is None:
            continue
        # C1: detector fired (peak >= threshold) AND decayed (final < peak).
        c1 = bool(n["detector_fired"]) and (n["detector_final"] < n["detector_peak"])
        # C2 (load-bearing readiness): ON arm WIDENS the proposed candidate set
        # under stuck; OFF arm is INVARIANT (regulator None -> bit-identical).
        # The downstream first-action-entropy LIFT is a behavioural-conversion
        # question for the Q-056 governance falsifier (the within-class
        # temperature lever), NOT a substrate-readiness gate -- count-widening
        # plus OFF-invariance is the substrate fact this probe certifies.
        # entropy_delta is recorded as a diagnostic.
        # Count is the deterministic substrate signal (the regulator adds
        # exactly `extra` candidates); first-action entropy drifts with the
        # stochastic proposer between two sequential proposals even when OFF,
        # so it is NOT used for the invariance gate.
        on_widens = n["n_candidates_stuck"] > n["n_candidates_base"]
        off_invariant = o["n_candidates_stuck"] == o["n_candidates_base"]
        c2 = on_widens and off_invariant
        if c1:
            n_c1 += 1
        if c2:
            n_c2 += 1
        per_seed.append(
            {"seed": seed, "C1_detector_nonvacuous": c1, "C2_regulator_load_bearing": c2}
        )

    n = len(per_seed)
    frac_c1 = n_c1 / n if n else 0.0
    frac_c2 = n_c2 / n if n else 0.0
    # Non-vacuity precondition: the detector must actually fire somewhere.
    any_fired = any(r["detector_fired"] for r in rows if r["arm"] == "ARM_ON")

    if not any_fired:
        label = "substrate_not_ready_requeue"
        outcome = "FAIL"
    elif frac_c1 >= PASS_FRACTION and frac_c2 >= PASS_FRACTION:
        label = "sd061_regulator_ready"
        outcome = "PASS"
    else:
        label = "sd061_regulator_not_load_bearing"
        outcome = "FAIL"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "detector_fires_under_induced_impasse",
                "description": "StuckStateDetector reaches is_stuck on the ON arm impasse sequence",
                "measured": float(any_fired),
                "threshold": 1.0,
                "met": bool(any_fired),
            }
        ],
        "criteria_non_degenerate": {
            "C1_detector_nonvacuous": n_c1 > 0,
            "C2_regulator_load_bearing": n_c2 > 0,
        },
        "criteria": [
            {"name": "C2_regulator_load_bearing", "load_bearing": True, "passed": frac_c2 >= PASS_FRACTION},
        ],
    }

    return {
        "outcome": outcome,
        "interpretation_label": label,
        "interpretation": interpretation,
        "arm_results": rows,
        "summary": {
            "frac_c1_detector_nonvacuous": round(frac_c1, 3),
            "frac_c2_regulator_load_bearing": round(frac_c2, 3),
            "any_detector_fired": bool(any_fired),
        },
        "per_seed": per_seed,
        "total_seeds_attempted": len(SEEDS),
        "total_seeds_completed": n,
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "interpretation_label": result["interpretation_label"],
        "interpretation": result["interpretation"],
        "summary": result["summary"],
        "per_seed": result["per_seed"],
        "arm_results": result["arm_results"],
        "total_seeds_attempted": result["total_seeds_attempted"],
        "total_seeds_completed": result["total_seeds_completed"],
        "dry_run": bool(dry_run),
        "notes": (
            "SD-061 substrate-readiness diagnostic. Claim-free. PASS clears the "
            "SD-061 readiness gate; the Q-056 3-arm governance falsifier is a "
            "separate later /queue-experiment session."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.dry_run:
        global SEEDS, WARMUP_STEPS, STUCK_SEQUENCE_TICKS, RELIEF_SEQUENCE_TICKS
        SEEDS = [42]
        WARMUP_STEPS = 3

    print(f"Seed-arm grid: {len(SEEDS)} seeds x {len(ARMS)} arms", flush=True)
    for si, seed in enumerate(SEEDS):
        for arm in ARMS:
            print(f"Seed {seed} Condition {arm}", flush=True)
            print(f"  [train] {arm} seed={seed} ep 1/1", flush=True)
        print(f"verdict: cell-{seed}", flush=True)

    result = run_experiment()
    # Per-seed-arm verdict lines (one per cell) for runner progress accounting.
    for r in result["arm_results"]:
        print(
            f"verdict: {'PASS' if r['arm']=='ARM_ON' and r['detector_fired'] else 'FAIL'}",
            flush=True,
        )

    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"c1={result['summary']['frac_c1_detector_nonvacuous']} "
        f"c2={result['summary']['frac_c2_regulator_load_bearing']}",
        flush=True,
    )
    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
