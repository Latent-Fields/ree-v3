"""V3-EXQ-738 -- Local-view-achievable ceiling anchor (competence-floor DIAGNOSTIC).

Establishes the LOCAL-VIEW-ACHIEVABLE ceiling anchor that the 732/732a competence-floor
discriminator was missing (WS-1 re-operationalization; failure_autopsy_V3-EXQ-732a_2026-07-10),
and by doing so DE-CONFOUNDS the H2 ("observation interface unlearnable") branch of the
post-724 competence-recovery campaign (siblings 734 env-difficulty / 735 drive-balance /
736 curriculum).

THE CONFOUND THIS CLOSES. The capability yardstick's only ceiling anchor is OraclePolicy,
which reads PRIVILEGED GLOBAL state (env.resources, all coords) and beelines from anywhere.
It proves the 1.0 resources/episode competence floor is achievable *with global information*,
but NOT that it is achievable from the same 5x5 partial observation the learner (REE, or a
vanilla-RL control) actually sees. That gap made the 732a sub-floor learner reading
uninterpretable: "the observation interface is unlearnable (H2)" vs "the yardstick is unfair"
(a global oracle vs a 5x5-local-view learner) could not be separated -- vanilla PPO scored
0.7 res/ep against a 0.5x-global-oracle bar of 28.6, structurally unreachable by ANY
local-view policy.

MECHANISM. This diagnostic adds LocalViewGreedyPolicy (experiments/_lib/capability_eval.py):
a one-step resource-gradient climber that reads ONLY obs_dict["resource_field_view"] -- the
agent-centered 5x5 gradient that is a subset of world_state, i.e. exactly what the REE encoder
senses. If that trivial local-view policy clears the 1.0 floor, then the floor is reachable
FROM THE LOCAL VIEW, the observation interface carries enough foraging signal, and a same-obs
learner that stays sub-floor is genuinely UNDER-POWERED (H1: action-learning stack), not
observation-starved (H2). It runs three policies -- local_view_greedy, greedy_oracle (global
ceiling / floor-achievability control, demoted per WS-1), random_walk (floor) -- across the
IDENTICAL 734 D0..D3 difficulty rungs, with NO training and NO REE mechanism, so it is a pure
achievability measurement.

PURPOSE: diagnostic (claim_ids=[]); promotes/demotes nothing; brake-EXEMPT (a competence
achievability anchor, not a conversion/de-commit falsifier; claim_ids=[] zeroes the re-derive
brake). It is a MEASUREMENT that hardens 734's H2 verdict -- a 734 "learner_or_observability_
ceiling" result is only safe to build an observation-encoding substrate on if THIS anchor
shows the local view is NOT achievable; if the anchor IS achievable, the correct build target
is the action-learning stack (H1 / V3-EXQ-737), not the observation encoding.

PRE-REGISTERED SELF-ROUTE (HYPOTHESIS, not a verdict -- adjudicate before any governance use):
  * READINESS fails (D3 global oracle below the floor on the hazard-free env, OR the
    resource_field_view channel is absent [use_proxy_fields=False]) -> the floor is not
    oracle-achievable / the local channel does not exist here, so NO achievability conclusion
    is licensed -> `substrate_not_ready_requeue`, NEVER an obs-ceiling verdict.
  * LOCAL-VIEW ACHIEVABLE: readiness holds AND local_view_greedy clears the floor on a strict
    majority of seeds at the hazard-free D3 rung -> `local_view_achievable`. The 5x5 local
    view carries enough signal for a trivial policy to forage competently; a same-obs learner
    that stays sub-floor is under-powered (points at H1 / the action-learning stack), and a
    734 obs-ceiling verdict would be a FALSE positive.
  * LOCAL-VIEW OBS CEILING: readiness holds (global oracle clears D3) but local_view_greedy
    stays sub-floor at D3 -> `local_view_obs_ceiling`. The floor is reachable with global info
    but NOT from the 5x5 local view -> genuine observation-interface insufficiency; this is the
    only reading that licenses a 734 obs-encoding build target (supports H2).

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- repo path bootstrap (mirror 734/724) ------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    OraclePolicy,
    RandomPolicy,
    rollout_episode,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
from experiments.v3_exq_734_env_difficulty_competence_recovery_sweep import (  # noqa: E402
    DIFFICULTY_RUNGS,
    D0_RUNG_ID,
    D3_RUNG_ID,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_738_local_view_achievable_ceiling_anchor"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Pure-eval achievability anchor: NO training, NO substrate learning, nothing to bank for
# arm-reuse (each policy is a pure function of (env, seed)). arm_fingerprint reuse machinery
# targets skipping expensive TRAINING re-runs; there is no training here.
ARM_FINGERPRINT_EXEMPT = "pure-eval achievability anchor: no training/substrate, nothing to bank for reuse"

SEEDS: List[int] = [42, 43, 44]
EVAL_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20

# Policy factories (name -> callable(seed) -> Policy). local_view_greedy + random_walk are
# seeded; greedy_oracle is deterministic.
POLICY_ORDER = ["local_view_greedy", "greedy_oracle", "random_walk"]


def _make_env(rung: Dict[str, Any], seed: int) -> CausalGridWorldV2:
    kw = dict(x724.ENV_KWARGS)
    kw.update(rung.get("overrides", {}))
    return CausalGridWorldV2(seed=seed, **kw)


def _make_policy(name: str, seed: int):
    if name == "local_view_greedy":
        return LocalViewGreedyPolicy(seed=seed)
    if name == "greedy_oracle":
        return OraclePolicy()
    if name == "random_walk":
        return RandomPolicy(seed=seed)
    raise ValueError(f"unknown policy {name}")


def _eval_cell(
    policy_name: str,
    rung: Dict[str, Any],
    seed: int,
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Run one (policy, rung, seed) cell; return foraging + channel-presence."""
    env = _make_env(rung, seed)
    policy = _make_policy(policy_name, seed)
    resources_per_ep: List[int] = []
    channel_present = False
    for ep in range(int(eval_episodes)):
        _flat, obs_dict = env.reset()
        if ep == 0:
            channel_present = obs_dict.get("resource_field_view") is not None
        policy.reset(env)
        row = rollout_episode(env, obs_dict, policy, steps_per_episode)
        resources_per_ep.append(int(row["resources"]))
        if (ep + 1) % 5 == 0 or (ep + 1) == eval_episodes:
            print(
                f"  [eval] {policy_name} rung={rung['rung_id']} seed={seed} "
                f"ep {ep + 1}/{eval_episodes}",
                flush=True,
            )
    n = len(resources_per_ep)
    foraging = float(sum(resources_per_ep) / n) if n else 0.0
    return {
        "policy": policy_name,
        "rung_id": rung["rung_id"],
        "seed": int(seed),
        "n_episodes": int(n),
        "foraging_competence": round(foraging, 6),
        "resource_field_view_present": bool(channel_present),
        "competence_supra_floor": bool(foraging >= COMPETENCE_RESOURCE_FLOOR),
        "per_episode_resources": [int(r) for r in resources_per_ep],
    }


def run_experiment(
    seeds: List[int],
    eval_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    # cells[rung_id][policy] = list of per-seed rows
    per_rung: Dict[str, Dict[str, Any]] = {}
    channel_present_all = True
    for rung in DIFFICULTY_RUNGS:
        rid = rung["rung_id"]
        per_rung[rid] = {}
        for policy_name in POLICY_ORDER:
            seed_rows: List[Dict[str, Any]] = []
            for seed in seeds:
                # Progress: boundary line resets episodes_in_run for the runner.
                print(
                    f"Seed {seed} Condition {rid}:{policy_name}",
                    flush=True,
                )
                row = _eval_cell(policy_name, rung, seed, eval_episodes, steps_per_episode)
                if not row["resource_field_view_present"]:
                    channel_present_all = False
                seed_rows.append(row)
                # One verdict line per (seed x condition) cell for runner run-accounting.
                print(
                    f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'}",
                    flush=True,
                )
            foragings = [r["foraging_competence"] for r in seed_rows]
            n_supra = int(sum(1 for r in seed_rows if r["competence_supra_floor"]))
            n_seeds = len(seed_rows)
            per_rung[rid][policy_name] = {
                "foraging_competence_mean": round(float(sum(foragings) / n_seeds), 6) if n_seeds else 0.0,
                "foraging_competence_per_seed": [round(f, 6) for f in foragings],
                "n_seeds": n_seeds,
                "n_seeds_supra_floor": n_supra,
                "majority_supra_floor": bool(n_supra >= (n_seeds + 1) // 2) if n_seeds else False,
            }

    def _forage(rid: str, pol: str) -> float:
        return float(per_rung[rid][pol]["foraging_competence_mean"])

    def _majority(rid: str, pol: str) -> bool:
        return bool(per_rung[rid][pol]["majority_supra_floor"])

    # ---- readiness gates -----------------------------------------------------
    d3_oracle_forage = _forage(D3_RUNG_ID, "greedy_oracle")
    d3_oracle_clears = bool(d3_oracle_forage >= COMPETENCE_RESOURCE_FLOOR)
    readiness_met = bool(d3_oracle_clears and channel_present_all)

    # ---- load-bearing criterion: local-view greedy clears the floor at D3 ----
    local_view_clears_d3 = _majority(D3_RUNG_ID, "local_view_greedy")

    # ---- non-degeneracy: greedy must beat the random floor at D3 (else the -----
    #      "achievable" reading is vacuous -- e.g. respawn makes random clear too).
    d3_greedy_forage = _forage(D3_RUNG_ID, "local_view_greedy")
    d3_random_forage = _forage(D3_RUNG_ID, "random_walk")
    greedy_beats_random_d3 = bool(d3_greedy_forage > d3_random_forage)
    random_below_floor_d3 = bool(d3_random_forage < COMPETENCE_RESOURCE_FLOOR)

    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif local_view_clears_d3:
        outcome = "PASS"
        label = "local_view_achievable"
    else:
        outcome = "FAIL"
        label = "local_view_obs_ceiling"

    interpretation = {
        "label": label,
        "preconditions": [
            {
                "name": "d3_global_oracle_clears_floor",
                "description": (
                    "The hazard-free D3 env must be floor-achievable with global info "
                    "(else the floor is unreachable here and no achievability read is licensed)."
                ),
                "control": "greedy_oracle (global) on the D3 hazard-free rung vs the 1.0 floor",
                "measured": round(d3_oracle_forage, 6),
                "threshold": float(COMPETENCE_RESOURCE_FLOOR),
                "met": bool(d3_oracle_clears),
            },
            {
                "name": "resource_field_view_channel_present",
                "description": (
                    "obs_dict must expose resource_field_view (use_proxy_fields=True) for a "
                    "local-view forager to exist; absent -> the anchor cannot be measured."
                ),
                "control": "obs_dict.get('resource_field_view') is not None on every cell",
                "measured": 1.0 if channel_present_all else 0.0,
                "threshold": 1.0,
                "met": bool(channel_present_all),
            },
        ],
        "criteria_non_degenerate": {
            "local_view_greedy_beats_random_floor_at_d3": greedy_beats_random_d3,
            "random_floor_below_competence_floor_at_d3": random_below_floor_d3,
            "d3_derisked_from_d0": bool(_forage(D3_RUNG_ID, "greedy_oracle") > _forage(D0_RUNG_ID, "greedy_oracle")),
        },
        "criteria": [
            {
                "name": "C_local_view_greedy_clears_floor_at_D3",
                "load_bearing": True,
                "passed": bool(local_view_clears_d3),
            }
        ],
    }

    return {
        "outcome": outcome,
        "interpretation": interpretation,
        "per_rung": per_rung,
        "readiness": {
            "readiness_met": readiness_met,
            "d3_global_oracle_forage": round(d3_oracle_forage, 6),
            "d3_global_oracle_clears_floor": d3_oracle_clears,
            "resource_field_view_channel_present": channel_present_all,
        },
        "headline": {
            "d3_local_view_greedy_forage": round(d3_greedy_forage, 6),
            "d3_local_view_greedy_clears_floor_majority": local_view_clears_d3,
            "d3_global_oracle_forage": round(d3_oracle_forage, 6),
            "d3_random_forage": round(d3_random_forage, 6),
            "d0_local_view_greedy_forage": round(_forage(D0_RUNG_ID, "local_view_greedy"), 6),
        },
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": [],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "brake_exempt": True,
        "brake_exempt_reason": "competence achievability anchor; claim_ids=[]; not a conversion/de-commit falsifier",
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation"]["label"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "per_rung": result["per_rung"],
        "config": {
            "seeds": SEEDS if not dry_run else DRY_RUN_SEEDS,
            "eval_episodes": EVAL_EPISODES if not dry_run else DRY_RUN_EVAL,
            "steps_per_episode": STEPS_PER_EPISODE if not dry_run else DRY_RUN_STEPS,
            "rungs": [r["rung_id"] for r in DIFFICULTY_RUNGS],
            "policies": POLICY_ORDER,
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        },
        "load_bearing_dv": (
            "D3 local_view_greedy mean resources/ep (obs_dict['resource_field_view'] only) "
            "vs the 1.0 competence floor, majority of seeds"
        ),
        "notes": (
            "Local-view-achievable ceiling anchor (WS-1 re-op). local_view_greedy reads ONLY "
            "the agent's 5x5 resource_field_view -- a subset of world_state the REE encoder "
            "already senses. A PASS (local_view_achievable) means the observation interface "
            "carries enough signal for a trivial policy to clear the floor, so a same-obs "
            "learner that stays sub-floor is under-powered (H1), not obs-starved (H2). "
            "De-confounds the 734 'learner_or_observability_ceiling' branch. Brake-exempt; "
            "PROMOTES/DEMOTES NOTHING."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description=(
            "V3-EXQ-738 local-view-achievable ceiling anchor DIAGNOSTIC "
            "(is the 1.0 competence floor reachable from the 5x5 local view; claim_ids=[])"
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        eval_eps = DRY_RUN_EVAL
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        eval_eps = EVAL_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, eval_episodes=eval_eps, steps_per_episode=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    hl = result["headline"]
    rd = result["readiness"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation']['label']} "
        f"readiness_met={rd['readiness_met']} "
        f"d3_local_view_greedy/ep={hl['d3_local_view_greedy_forage']} "
        f"(clears_floor_majority={hl['d3_local_view_greedy_clears_floor_majority']}) "
        f"d3_oracle/ep={hl['d3_global_oracle_forage']} d3_random/ep={hl['d3_random_forage']}",
        flush=True,
    )
    for rung in DIFFICULTY_RUNGS:
        rid = rung["rung_id"]
        pr = result["per_rung"][rid]
        print(
            f"  RUNG {rid}: local_view_greedy/ep={pr['local_view_greedy']['foraging_competence_mean']} "
            f"(supra {pr['local_view_greedy']['n_seeds_supra_floor']}/{pr['local_view_greedy']['n_seeds']}) "
            f"oracle/ep={pr['greedy_oracle']['foraging_competence_mean']} "
            f"random/ep={pr['random_walk']['foraging_competence_mean']}",
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
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
