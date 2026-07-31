"""V3-EXQ-852 -- SD-085 substrate-readiness diagnostic: E3Config.f_weight.

PURPOSE (diagnostic / substrate-readiness validation, NOT governance evidence,
NOT ARC-062 GOV-FANOUT-1 Leg P-B). SD-085 (REE_assembly
docs/architecture/sd_085_e3_reality_cost_weight.md) added a no-op-default
E3Config.f_weight coefficient scaling F's contribution to
E3TrajectorySelector.score_trajectory's committed-selection score. A unit-level
smoke test (real E3TrajectorySelector + real Trajectory, tensor-exact
assertions) already confirmed the knob at the score_trajectory function level
during implementation. This run is the substrate-readiness step Step 8 of
/implement-substrate requires: confirm the knob is reachable and behaves
correctly through the FULL AGENT pipeline (REEConfig.from_dims -> REEAgent ->
agent.e3.select()) on a real env, and stays numerically stable over a short
run. Leg P-B itself (the GOV-FANOUT-1 F-dominance scientific ladder, many
seeds, rule-apprehension channel ON, C1/C2 statistics) is separate, deliberately
out of scope here, and follows in its own /queue-experiment session.

GOV-REUSE-1 (Step 2.4): the decisive readout is E3Config.f_weight's effect on
score_trajectory's output through a real agent's e3.select() call. f_weight did
not exist before this session (SD-085 landed 2026-07-31), so no recorded
manifest on any substrate_hash can carry it. Not recoverable -> run.

ARMS (2, same seeds): ARM_DEFAULT f_weight=1.0 (must reproduce the exact
pre-SD-085 score formula) vs ARM_ATTENUATED f_weight=0.5 (must measurably
change the score). Both arms otherwise identical (same env config, same
torch-seeded network weights, use_candidate_rule_field/routing all left at
their substrate defaults -- this run is NOT the rule-apprehension-channel-ON
configuration Leg P-B needs).

Direct e3.select(...) call at every tick (never through the cadence-gated
agent.select_action() wrapper) -- this is the documented exemption from the
E3-diagnostics-staleness pseudo-replication hazard (validate_experiments.py
e3_diagnostics_staleness check): every recorded row corresponds to a genuine
select() call, never a latched read.

DV-SYMMETRY (Step 3 mandatory declaration, per arm): DV = per-candidate score
tensor from e3.select(...).scores. Symmetry group: permutation of candidate
slots (scores are computed independently per candidate, order-preserving).
The manipulation (a per-arm CONSTANT multiplicative factor 0.5 vs 1.0 applied
to F, one of three additive terms in the score) is NOT invariant under any
symmetry of this DV that would erase it: it is not a broadcast constant added
uniformly across candidates (it scales, term-by-term, a component that already
varies per candidate), so it changes each candidate's score by a
candidate-dependent amount (0.5 * f_i, and f_i is not constant across
candidates in a live agent) -- not a uniform shift that argmax/softmax would
cancel.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402

_ZG = ZGoalStreamAccumulator()

EXPERIMENT_TYPE = "v3_exq_852_sd085_f_weight_substrate_readiness"
EXPERIMENT_PURPOSE = "diagnostic"
ANCHOR_REACHABILITY_EXEMPT = (
    "candidate_pool_non_trivial IS the degeneracy definition, not a narrower "
    "hand-written proxy: CausalGridWorldV2's candidate generation at this env "
    "config deterministically yields >1 candidate every tick by construction "
    "(the deterministic action set), so the precondition is reachable by "
    "construction and is not a separate control that could fail independently "
    "of the quantity it asserts."
)
CLAIM_IDS: List[str] = []

SEEDS = [11, 22, 33]
ARMS = ["ARM_DEFAULT", "ARM_ATTENUATED"]
N_EPISODES = 2
STEPS_PER_EPISODE = 15
F_WEIGHT_BY_ARM = {"ARM_DEFAULT": 1.0, "ARM_ATTENUATED": 0.5}

# Pre-registered thresholds (defined here, never inferred post-hoc).
DIFF_FLOOR = 1e-6           # C1: attenuated tick-0 score must differ from default by more
DEFAULT_MATCH_TOL = 0.0     # C2: default-arm score must bit-match the pre-SD-085 formula


def _build_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=6, num_hazards=1, num_resources=4,
                             use_proxy_fields=True, seed=seed)


def _config_slice(f_weight: float) -> Dict[str, Any]:
    return {
        "env": {"size": 6, "num_hazards": 1, "num_resources": 4,
                "use_proxy_fields": True},
        "schedule": {"n_episodes": N_EPISODES, "steps": STEPS_PER_EPISODE},
        "f_weight": f_weight,
    }


def _build_agent(env: CausalGridWorldV2, seed: int) -> REEAgent:
    torch.manual_seed(seed)  # identical network init weights across arms at a given seed
    agent = REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
    ))
    agent.e3.e3_score_decomp_enabled = True
    return agent


def _run_cell(arm: str, seed: int) -> Dict[str, Any]:
    f_weight = F_WEIGHT_BY_ARM[arm]
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(f_weight),
                  script_path=Path(__file__)) as cell:
        env = _build_env(seed)
        agent = _build_agent(env, seed)
        agent.e3.config.f_weight = f_weight  # SD-085 lever: set directly, matches
                                              # the lambda_ethical/rho_residue sweep idiom

        f_weighted_samples: List[float] = []
        raw_score_samples: List[float] = []
        n_ticks = 0
        n_finite_violations = 0
        tick0_scores = None
        tick0_f_raw = None
        tick0_f_weighted = None

        for ep in range(N_EPISODES):
            _, obs = env.reset()
            agent.reset()
            for step_idx in range(STEPS_PER_EPISODE):
                latent = agent.sense(obs["body_state"], obs["world_state"])
                ticks = agent.clock.advance()
                wd = agent.config.latent.world_dim
                e1 = (agent._e1_tick(latent) if ticks.get("e1_tick")
                      else torch.zeros(1, wd, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1, ticks)

                # Direct e3.select() call every tick -- never through the
                # cadence-gated select_action() wrapper, so every row here is a
                # genuine fresh selection (no latch pseudo-replication).
                result = agent.e3.select(candidates, temperature=1.0)
                scores = result.scores.detach()
                decomp = agent.e3._last_traj_components

                n_ticks += 1
                finite = bool(torch.isfinite(scores).all())
                if not finite:
                    n_finite_violations += 1
                raw_score_samples.append(float(scores.mean().item()))
                if decomp is not None:
                    f_weighted_samples.append(decomp["f_weighted"])
                if ep == 0 and step_idx == 0:
                    tick0_scores = scores.clone()
                    tick0_f_raw = decomp["f"] if decomp is not None else None
                    tick0_f_weighted = decomp["f_weighted"] if decomp is not None else None

                action = agent.select_action(candidates, ticks)
                _, _h, done, _, obs = env.step(int(action.argmax(dim=-1).item()))
                if done:
                    break
            print(f"  [train] f_weight seed={seed} arm={arm} ep {ep+1}/{N_EPISODES} "
                  f"ticks={n_ticks}", flush=True)

        _ZG.observe(agent)

        row = {
            "arm": arm,
            "seed": seed,
            "f_weight": f_weight,
            "n_ticks": n_ticks,
            "n_finite_violations": n_finite_violations,
            "score_mean": statistics.fmean(raw_score_samples) if raw_score_samples else None,
            "f_weighted_mean": (statistics.fmean(f_weighted_samples)
                                 if f_weighted_samples else None),
            "tick0_score_mean": float(tick0_scores.mean().item()) if tick0_scores is not None else None,
            "tick0_score_vec": [float(x) for x in tick0_scores.tolist()] if tick0_scores is not None else None,
            "tick0_f_raw": tick0_f_raw,
            "tick0_f_weighted": tick0_f_weighted,
            "n_candidates_tick0": int(tick0_scores.shape[0]) if tick0_scores is not None else 0,
        }
        cell.stamp(row)
    passed = (row["n_finite_violations"] == 0) and (row["n_ticks"] > 0)
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def run_experiment() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            rows.append(_run_cell(arm, seed))

    default_rows = {r["seed"]: r for r in rows if r["arm"] == "ARM_DEFAULT"}
    attenuated_rows = {r["seed"]: r for r in rows if r["arm"] == "ARM_ATTENUATED"}

    # READINESS precondition: tick-0 candidate pool must be non-trivial (>1
    # candidate) for a per-candidate score comparison to mean anything.
    min_candidates = min((r["n_candidates_tick0"] for r in rows), default=0)
    candidates_present = min_candidates > 1

    # C1: attenuated tick-0 mean score differs measurably from default, per seed.
    diffs = []
    for seed in SEEDS:
        d = default_rows[seed]["tick0_score_mean"]
        a = attenuated_rows[seed]["tick0_score_mean"]
        if d is None or a is None:
            diffs.append(None)
        else:
            diffs.append(abs(d - a))
    c1_per_seed = [bool(x is not None and x > DIFF_FLOOR) for x in diffs]
    c1 = all(c1_per_seed) if candidates_present else False

    # C2: default-arm tick-0 f_weighted decomp value equals tick-0 raw f
    # (f_weight=1.0 is a no-op) to within float tolerance -- confirms the knob
    # threads through the real agent pipeline bit-identically at default, not
    # just in isolation. Compares tick-0 to tick-0 (both quantities are read
    # from the SAME select() call), never a raw value against a multi-tick mean.
    c2_per_seed = []
    for seed in SEEDS:
        r = default_rows[seed]
        if r["tick0_f_raw"] is None or r["tick0_f_weighted"] is None:
            c2_per_seed.append(False)
            continue
        c2_per_seed.append(abs(r["tick0_f_weighted"] - r["tick0_f_raw"]) <= DEFAULT_MATCH_TOL + 1e-9)
    c2 = all(c2_per_seed)

    # C3: numerically stable -- zero finite violations across every cell.
    c3 = all(r["n_finite_violations"] == 0 for r in rows)

    overall = c1 and c2 and c3 and candidates_present
    if not candidates_present:
        label = "substrate_not_ready_requeue"
    elif overall:
        label = "sd085_f_weight_reachable_through_full_agent_pipeline"
    else:
        label = "sd085_f_weight_substrate_defect"

    metrics = {
        "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,
        "c1_per_seed": c1_per_seed, "c2_per_seed": c2_per_seed,
        "tick0_score_diffs_default_vs_attenuated": diffs,
        "min_candidates_tick0": min_candidates,
        "n_ticks_total": sum(r["n_ticks"] for r in rows),
        "n_finite_violations_total": sum(r["n_finite_violations"] for r in rows),
    }
    return {
        "outcome": "PASS" if overall else "FAIL",
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": [
                {"name": "candidate_pool_non_trivial",
                 "description": ("tick-0 candidate pool must hold >1 candidate for a "
                                  "per-candidate score comparison to be meaningful."),
                 "measured": float(min_candidates), "threshold": 1.0,
                 "direction": "lower",
                 "control": "worst (minimum) candidate count across all cells at tick 0",
                 "met": candidates_present},
            ],
            "criteria": [
                {"name": "C1_attenuated_score_differs", "load_bearing": True, "passed": c1},
                {"name": "C2_default_arm_bit_identical_to_raw_f", "load_bearing": True, "passed": c2},
                {"name": "C3_numerically_stable", "load_bearing": True, "passed": c3},
            ],
            "criteria_non_degenerate": {
                "C1_attenuated_score_differs": bool(candidates_present),
                "C2_default_arm_bit_identical_to_raw_f": bool(candidates_present),
                "C3_numerically_stable": True,
            },
            "note": ("Does NOT test ARC-062 GOV-FANOUT-1 Leg P-B's scientific question "
                     "(rule-apprehension channel ON, F-attenuation ladder, committed-class "
                     "entropy). This run only confirms f_weight is reachable and correct "
                     "through the full agent pipeline at default substrate config."),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, N_EPISODES
    if args.dry_run:
        SEEDS = [11]
        N_EPISODES = 1

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS, "arms": ARMS, "n_episodes": N_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "diff_floor": DIFF_FLOOR,
        "f_weight_by_arm": F_WEIGHT_BY_ARM,
        "arm_config_slice_default": _config_slice(1.0),
        "arm_config_slice_attenuated": _config_slice(0.5),
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    m = result["metrics"]
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(f"C1={m['c1_pass']} C2={m['c2_pass']} C3={m['c3_pass']} "
          f"min_candidates={m['min_candidates_tick0']}", flush=True)
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
