#!/opt/local/bin/python3
"""V3-EXQ-748 -- MECH-457 GOV-FANOUT-1 leg B (H-explore) -- z_world dense-teacher DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending.

WHY. V3-EXQ-742 refuted "single missing mechanism = MECH-457": all four RPE actor-critic arms
forage 0.20-0.27/ep at D3 with every readiness precondition met, and train forage never left
~1.2 (the policy-gradient never bootstrapped from the sparse foraging reward). The 742 autopsy
routed the surviving deficit to a GOV-FANOUT-1 2x2 discrimination portfolio. This leg is the
DRIVE / REWARD-DENSITY axis.

THE 2x2 FACTORIAL (742 = the (z_world, sparse) cell):
                     sparse foraging RL        dense teacher (shaping + BC)
    z_world  (R0)    742: FAIL (cited)         THIS (V3-EXQ-748, H-explore)
    raw 5x5  (R1)    V3-EXQ-747 (H-rep)        V3-EXQ-749 (conjunction)

H-explore HYPOTHESIS. The sparse foraging reward could not bootstrap the policy-gradient
(reward-sparsity / exploration is the wall). TEST: KEEP z_world as the actor's input (742's
path via agent.actor_critic_step, cotrain_encoder=True -- 742's most-favorable arm) but replace
the sparse teacher with a DENSE one, in two sibling instantiations:
  * ac_zworld_shaped_rl -- 742 foraging reward + potential-based distance-to-nearest-resource
    shaping (Ng et al. 1999 policy-invariant Phi(s) = -manhattan_to_nearest_resource).
  * ac_zworld_bc        -- SUPERVISED behavior-cloning of LocalViewGreedyPolicy through the
    z_world path (no RL). A FAILED CE fit here is the direct "prediction-trained z_world is
    action-inadequate" signal: the expert reads the raw view, the actor must reproduce its
    action from z_world (recorded as bc_action_match_accuracy).

DECLARED NULL (so a sub-floor leg is informative, not wasted):
  * either dense arm clears the 1.0 floor -> reward-sparsity / exploration was the wall
    (a denser signal bootstraps the z_world policy). SELF-ROUTE:
    dense_teacher_on_zworld_clears_sparsity_was_the_wall.
  * both dense arms sub-floor -> sparsity is NOT the sole wall; a dense teacher on the z_world
    path still cannot forage -> z_world may be action-inadequate (the conjunction leg 749, raw
    view + dense teacher, decides). SELF-ROUTE: dense_teacher_on_zworld_insufficient.

READINESS (P0 readiness-assert; same statistic as the verdict). LocalViewGreedyPolicy (5x5 view)
and greedy_oracle clear the 1.0 floor @D3 (env solvable). Below readiness ->
substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict label); treatment training skipped.

evidence_direction = "unknown" (DIAGNOSTIC; the discrimination verdict lives in
interpretation.label / discrimination_verdict, adjudicated by /failure-autopsy).

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech457_fanout.py. ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import COMPETENCE_RESOURCE_FLOOR, evaluate_seed  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_748_mech457_hexplore_reward_density_actor_critic"
QUEUE_ID = "V3-EXQ-748"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

TREATMENT_ARMS: Tuple[str, ...] = ("ac_zworld_shaped_rl", "ac_zworld_bc")
ARM_ORDER: Tuple[str, ...] = TREATMENT_ARMS + fan.ANCHOR_ARMS


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], p0: int, rl_eps: int,
                      bc_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    if arm_id in TREATMENT_ARMS:
        base.update({
            "kind": "zworld_actor_critic", "representation": "z_world_cotrain",
            "cotrain_encoder": True, "use_sf_critic": False,
            "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
            "p0_warmup_episodes": int(p0),
        })
        if arm_id == "ac_zworld_shaped_rl":
            base.update({"teacher": "foraging_plus_potential_shaping",
                         "shaping_coef": fan.SHAPING_COEF, "rl_episodes": int(rl_eps)})
        else:
            base.update({"teacher": "behavior_cloning_local_view_greedy",
                         "bc_episodes": int(bc_eps)})
    else:
        base.update({"kind": "anchor"})
    return base


def _run_treatment_cell(arm_id: str, env_kwargs: Dict[str, Any], seed: int, p0: int,
                        rl_eps: int, bc_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    warm_env = x734._make_env(seed, env_kwargs)
    agent = fan.make_zworld_agent(warm_env)
    fan.warmup_zworld(agent, warm_env, seed=seed, p0=p0, steps=steps)

    train_env = x734._make_env(seed, env_kwargs)
    extra: Dict[str, Any] = {}
    if arm_id == "ac_zworld_shaped_rl":
        guard = fan.train_zworld_ac_shaped(
            agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=fan.SHAPING_COEF,
        )
        extra["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    else:  # ac_zworld_bc
        guard = fan.bc_warmup_zworld(
            agent, train_env, seed=seed, n_bc=bc_eps, steps=steps,
            arm_label=arm_id, denom=bc_eps,
        )
        extra["bc_action_match_accuracy_recent"] = guard["bc_action_match_accuracy_recent"]

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(x742.ActorCriticEvalPolicy(agent, arm_id), eval_env, eval_eps, steps)
    row.update(extra)
    return row


def run_experiment(seeds: List[int], p0: int, rl_eps: int, bc_eps: int,
                   eval_eps: int, steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg B (H-explore z_world dense-teacher) diagnostic "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, RL={rl_eps}, BC={bc_eps}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_bc_acc: Dict[str, List[float]] = {"ac_zworld_bc": []}
    per_arm_trainforage: Dict[str, List[float]] = {"ac_zworld_shaped_rl": []}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, p0, rl_eps, bc_eps, eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in TREATMENT_ARMS:
                row = _run_treatment_cell(arm_id, env_kwargs, seed, p0, rl_eps, bc_eps, eval_eps, steps)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id == "ac_zworld_shaped_rl":
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        if arm_id == "ac_zworld_bc":
            per_arm_bc_acc[arm_id].append(float(row.get("bc_action_match_accuracy_recent", 0.0)))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    readiness_met = bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )

    if readiness_met:
        for arm_id in TREATMENT_ARMS:
            for seed in seeds:
                _run_cell(arm_id, seed)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping treatment training -> substrate_not_ready_requeue", flush=True,
        )

    shaped = fan.summarize(per_arm_forage["ac_zworld_shaped_rl"])
    bc = fan.summarize(per_arm_forage["ac_zworld_bc"])
    shaped_maj = bool(shaped["majority_supra_floor"])
    bc_maj = bool(bc["majority_supra_floor"])
    any_dense_maj = bool(shaped_maj or bc_maj)

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_dense_maj:
        outcome, label = "PASS", "dense_teacher_on_zworld_clears_sparsity_was_the_wall"
    else:
        outcome, label = "FAIL", "dense_teacher_on_zworld_insufficient"

    degeneracy = check_degeneracy({
        "d3_zworld_dense_vs_anchor_foraging": {
            "values": [shaped["foraging_competence_mean"], bc["foraging_competence_mean"],
                       local_view_mean, _mean("random_walk")]
        }
    })

    tf = per_arm_trainforage["ac_zworld_shaped_rl"]
    shaped_train_mean = round(float(sum(tf) / len(tf)), 6) if tf else 0.0
    bc_acc = per_arm_bc_acc["ac_zworld_bc"]
    bc_acc_mean = round(float(sum(bc_acc) / len(bc_acc)), 6) if bc_acc else 0.0

    interpretation = {
        "label": label,
        "preconditions": [
            fan.readiness_precondition(local_view_mean),
            {"name": "greedy_oracle_clears_floor_at_d3", "kind": "readiness",
             "description": "Env is floor-achievable with global info (achievability anchor).",
             "control": "greedy_oracle foraging_competence @D3 vs the 1.0 floor",
             "measured": round(oracle_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
        ],
        "criteria": [
            {"name": "C_any_zworld_dense_arm_clears_floor_at_D3", "load_bearing": True,
             "passed": bool(any_dense_maj)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "zworld_dense_vs_anchor_foraging_spread": bool(degeneracy["non_degenerate"]),
        },
    }

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
        },
        "headline": {
            "d3_zworld_shaped_forage": shaped["foraging_competence_mean"],
            "d3_zworld_shaped_per_seed": shaped["foraging_competence_per_seed"],
            "d3_zworld_bc_forage": bc["foraging_competence_mean"],
            "d3_zworld_bc_per_seed": bc["foraging_competence_per_seed"],
            "d3_zworld_bc_action_match_accuracy": bc_acc_mean,
            "d3_any_dense_majority_supra_floor": any_dense_maj,
            "d3_shaped_majority_supra_floor": shaped_maj,
            "d3_bc_majority_supra_floor": bc_maj,
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "reference_742_zworld_sparse_forage_d3": "0.20-0.27 (cited, not re-run)",
        },
        "bootstrap_guard": {
            "load_bearing": False,
            "d3_shaped_mean_train_forage_recent": shaped_train_mean,
            "d3_shaped_eval_foraging": shaped["foraging_competence_mean"],
            "d3_bc_action_match_accuracy_recent": bc_acc_mean,
            "note": (
                "bc_action_match_accuracy is a DIRECT z_world action-adequacy readout: a low "
                "fit means the prediction-trained z_world cannot reproduce the competent "
                "expert's action (points to H-rep / the conjunction leg 749)."
            ),
        },
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in ARM_ORDER},
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "local_view_greedy_d3_live": round(local_view_mean, 6),
            "local_view_greedy_d3_738_reference": float(fan.DENOM_738_D3_REFERENCE),
        },
        "arm_results": all_cells,
        "non_degenerate": bool(degeneracy["non_degenerate"]),
        "degeneracy_reason": degeneracy["degeneracy_reason"],
        "degenerate_metrics": degeneracy["degenerate_metrics"],
    }
    return result


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "interpretation": result["interpretation"],
        "interpretation_label": result["interpretation_label"],
        "discrimination_verdict": result["discrimination_verdict"],
        "readiness": result["readiness"],
        "headline": result["headline"],
        "bootstrap_guard": result["bootstrap_guard"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "portfolio": {
            "gov_fanout_1": "MECH-457 competence-discrimination portfolio (742 autopsy)",
            "leg": "B (H-explore, drive / reward-density axis)",
            "factorial_cell": "(z_world_cotrain, dense_teacher: shaping + BC)",
            "siblings": ["V3-EXQ-747 (H-rep)", "V3-EXQ-749 (conjunction)"],
            "reference_cell_742": "(z_world, sparse) -> FAIL (cited, not re-run)",
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 z_world (cotrain) actor-critic foraging_competence under a DENSE teacher "
            "(shaping OR BC), unshaped eval, vs the 1.0 floor, strict majority of seeds; "
            "readiness = local_view_greedy + oracle clear the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg B. DIAGNOSTIC (excluded from scoring); PROMOTES/DEMOTES NOTHING; "
            "route to /failure-autopsy before any governance action. Same-question 742 floor "
            "re-pose REFUSED -- this attacks the REWARD-DENSITY axis (a different mechanism), "
            "not a power-bump of the sparse-RL design. MECH-457 stays candidate/v3_pending. "
            "GOV-REUSE-1: the decisive readout (z_world actor-critic foraging under a dense "
            "teacher) is recorded in NO prior manifest (742 was sparse only) -> run."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-748 MECH-457 H-explore z_world dense-teacher diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0, rl_eps, bc_eps = fan.DRY_P0, fan.DRY_RL, fan.DRY_BC
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0, rl_eps, bc_eps = fan.P0_WARMUP_EPISODES, fan.RL_EPISODES, fan.BC_EPISODES
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, rl_eps=rl_eps, bc_eps=bc_eps,
                            eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "p0_warmup_episodes": p0, "rl_episodes": rl_eps, "bc_episodes": bc_eps,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN, "cotrain_encoder": True,
        "ac_lr": fan.AC_LR, "bc_lr": fan.BC_LR, "ac_gamma": fan.AC_GAMMA,
        "shaping_coef": fan.SHAPING_COEF,
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    hl = result["headline"]
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"readiness_met={result['readiness']['readiness_met']} "
        f"non_degenerate={result['non_degenerate']}", flush=True,
    )
    print(
        f"  D3: shaped={hl['d3_zworld_shaped_forage']} bc={hl['d3_zworld_bc_forage']} "
        f"(bc_acc={hl['d3_zworld_bc_action_match_accuracy']}) "
        f"local_view={hl['d3_local_view_greedy_denominator']} "
        f"(any_dense_supra={hl['d3_any_dense_majority_supra_floor']})", flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = str(result["outcome"]).upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel, bool(args.dry_run)


if __name__ == "__main__":
    _outcome, _manifest_path, _dry_run = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path, dry_run=_dry_run)
