#!/opt/local/bin/python3
"""V3-EXQ-749 -- MECH-457 GOV-FANOUT-1 leg C (H-rep x H-explore CONJUNCTION) DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending.

WHY THIS LEG EXISTS (the /queue-experiment Step 2.5b design-audit output). The 742 autopsy
named two live hypotheses -- H-rep (V3-EXQ-747: raw view, sparse) and H-explore (V3-EXQ-748:
z_world, dense). Those two single-axis legs share a VERDICT-ALIASING gap: if BOTH fail, they
cannot distinguish
  (a) "deeper than both -- even an action-adequate input AND a dense teacher jointly won't
      clear" (-> H-optim / capacity / credit-assignment), from
  (b) "the deficit is the CONJUNCTION -- neither axis alone suffices but both together do"
      (-> build an action-adequate encoder AND a dense teacher).
Two distinct causes -> the same reading (747 FAIL + 748 FAIL). This leg fills the empty (raw
view, dense) cell of the 2x2 factorial and de-aliases every verdict.

THE 2x2 FACTORIAL (742 = the (z_world, sparse) cell):
                     sparse foraging RL        dense teacher (shaping + BC)
    z_world  (R0)    742: FAIL (cited)         V3-EXQ-748 (H-explore)
    raw 5x5  (R1)    V3-EXQ-747 (H-rep)        THIS (V3-EXQ-749, conjunction)

Attribution once all cells land: 747 clears -> representation main effect; 748 clears -> reward
main effect; ONLY 749 clears -> interaction (need both); NONE clears -> deeper than both.

TEST. A standalone ActorCriticPolicy(world_dim=25) on the raw 5x5 resource_field_view (as in
747) AND a DENSE teacher (as in 748), two sibling instantiations:
  * ac_rawview_shaped_rl -- 742 foraging reward + potential-based distance-to-nearest-resource
    shaping (Ng et al. 1999 policy-invariant Phi(s) = -manhattan_to_nearest_resource).
  * ac_rawview_bc        -- SUPERVISED behavior-cloning of LocalViewGreedyPolicy from the raw
    view (no RL). Its cross-leg contrast with 748's ac_zworld_bc is decisive: raw-view BC
    clears while z_world BC does not => z_world is action-inadequate.

DECLARED NULL (so a sub-floor leg is informative, not wasted):
  * either arm clears the 1.0 floor -> the CONJUNCTION (action-adequate input + dense teacher)
    is what competence needs; neither axis alone was sufficient (given 747 / 748 sub-floor).
    SELF-ROUTE: conjunction_clears_need_adequate_input_and_dense_teacher.
  * both arms sub-floor -> even an action-adequate raw view WITH a dense teacher cannot clear
    the floor -> the deficit is DEEPER than representation + reward-sparsity together; escalate
    to the algorithm axis (H-optim: PPO/entropy/GAE) or a capacity / credit-assignment autopsy.
    DO NOT re-pose the floor. SELF-ROUTE: deeper_than_representation_and_reward_sparsity.

READINESS (P0 readiness-assert; same statistic as the verdict). LocalViewGreedyPolicy (the same
5x5 view) and greedy_oracle clear the 1.0 floor @D3 (env solvable from the local view). Below
readiness -> substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict label); treatment
training skipped.

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

EXPERIMENT_TYPE = "v3_exq_749_mech457_hrep_x_hexplore_rawview_dense_actor_critic"
QUEUE_ID = "V3-EXQ-749"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

TREATMENT_ARMS: Tuple[str, ...] = ("ac_rawview_shaped_rl", "ac_rawview_bc")
ARM_ORDER: Tuple[str, ...] = TREATMENT_ARMS + fan.ANCHOR_ARMS


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], rl_eps: int,
                      bc_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    if arm_id in TREATMENT_ARMS:
        base.update({
            "kind": "rawview_actor_critic", "representation": "raw_5x5_resource_field_view",
            "world_dim": fan.RAW_VIEW_DIM, "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
        })
        if arm_id == "ac_rawview_shaped_rl":
            base.update({"teacher": "foraging_plus_potential_shaping",
                         "shaping_coef": fan.SHAPING_COEF, "rl_episodes": int(rl_eps)})
        else:
            base.update({"teacher": "behavior_cloning_local_view_greedy",
                         "bc_episodes": int(bc_eps)})
    else:
        base.update({"kind": "anchor"})
    return base


def _run_treatment_cell(arm_id: str, env_kwargs: Dict[str, Any], seed: int, rl_eps: int,
                        bc_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    ac = fan.make_rawview_ac()
    train_env = x734._make_env(seed, env_kwargs)
    extra: Dict[str, Any] = {}
    if arm_id == "ac_rawview_shaped_rl":
        guard = fan.train_rawview_ac_rl(
            ac, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=fan.SHAPING_COEF,
        )
        extra["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    else:  # ac_rawview_bc
        guard = fan.bc_warmup_rawview(
            ac, train_env, seed=seed, n_bc=bc_eps, steps=steps, arm_label=arm_id, denom=bc_eps,
        )
        extra["bc_action_match_accuracy_recent"] = guard["bc_action_match_accuracy_recent"]

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(fan.RawViewACEvalPolicy(ac, arm_id), eval_env, eval_eps, steps)
    row.update(extra)
    return row


def run_experiment(seeds: List[int], rl_eps: int, bc_eps: int,
                   eval_eps: int, steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg C (H-rep x H-explore conjunction, raw-view dense) diagnostic "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"RL={rl_eps}, BC={bc_eps}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_bc_acc: Dict[str, List[float]] = {"ac_rawview_bc": []}
    per_arm_trainforage: Dict[str, List[float]] = {"ac_rawview_shaped_rl": []}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, rl_eps, bc_eps, eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in TREATMENT_ARMS:
                row = _run_treatment_cell(arm_id, env_kwargs, seed, rl_eps, bc_eps, eval_eps, steps)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id == "ac_rawview_shaped_rl":
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        if arm_id == "ac_rawview_bc":
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

    shaped = fan.summarize(per_arm_forage["ac_rawview_shaped_rl"])
    bc = fan.summarize(per_arm_forage["ac_rawview_bc"])
    shaped_maj = bool(shaped["majority_supra_floor"])
    bc_maj = bool(bc["majority_supra_floor"])
    any_maj = bool(shaped_maj or bc_maj)

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_maj:
        outcome, label = "PASS", "conjunction_clears_need_adequate_input_and_dense_teacher"
    else:
        outcome, label = "FAIL", "deeper_than_representation_and_reward_sparsity"

    degeneracy = check_degeneracy({
        "d3_rawview_dense_vs_anchor_foraging": {
            "values": [shaped["foraging_competence_mean"], bc["foraging_competence_mean"],
                       local_view_mean, _mean("random_walk")]
        }
    })

    tf = per_arm_trainforage["ac_rawview_shaped_rl"]
    shaped_train_mean = round(float(sum(tf) / len(tf)), 6) if tf else 0.0
    bc_acc = per_arm_bc_acc["ac_rawview_bc"]
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
            {"name": "C_any_rawview_dense_arm_clears_floor_at_D3", "load_bearing": True,
             "passed": bool(any_maj)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "rawview_dense_vs_anchor_foraging_spread": bool(degeneracy["non_degenerate"]),
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
            "d3_rawview_shaped_forage": shaped["foraging_competence_mean"],
            "d3_rawview_shaped_per_seed": shaped["foraging_competence_per_seed"],
            "d3_rawview_bc_forage": bc["foraging_competence_mean"],
            "d3_rawview_bc_per_seed": bc["foraging_competence_per_seed"],
            "d3_rawview_bc_action_match_accuracy": bc_acc_mean,
            "d3_any_majority_supra_floor": any_maj,
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
                "Cross-leg contrast: compare this raw-view bc_action_match_accuracy to leg B "
                "(V3-EXQ-748) ac_zworld_bc -- raw-view BC clears while z_world BC does not => "
                "z_world is action-inadequate."
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
            "leg": "C (H-rep x H-explore conjunction; the Step 2.5b verdict-aliasing closer)",
            "factorial_cell": "(raw_5x5_view, dense_teacher: shaping + BC)",
            "siblings": ["V3-EXQ-747 (H-rep)", "V3-EXQ-748 (H-explore)"],
            "reference_cell_742": "(z_world, sparse) -> FAIL (cited, not re-run)",
            "closes": "747-FAIL + 748-FAIL aliasing (deeper-than-both vs need-the-conjunction)",
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 raw-5x5-view actor-critic foraging_competence under a DENSE teacher (shaping OR "
            "BC), unshaped eval, vs the 1.0 floor, strict majority of seeds; readiness = "
            "local_view_greedy (same view) + oracle clear the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg C -- the design-audit (Step 2.5b) leg that closes the 747/748 "
            "verdict-aliasing gap. DIAGNOSTIC (excluded from scoring); PROMOTES/DEMOTES NOTHING; "
            "route to /failure-autopsy before any governance action. Same-question 742 floor "
            "re-pose REFUSED -- this is the (raw view, dense) factorial cell, a distinct "
            "mechanism combination, not a power-bump. MECH-457 stays candidate/v3_pending. "
            "GOV-REUSE-1: the (raw view, dense) readout is recorded in NO prior manifest -> run."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-749 MECH-457 H-rep x H-explore conjunction diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        rl_eps, bc_eps = fan.DRY_RL, fan.DRY_BC
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        rl_eps, bc_eps = fan.RL_EPISODES, fan.BC_EPISODES
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, rl_eps=rl_eps, bc_eps=bc_eps,
                            eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "rl_episodes": rl_eps, "bc_episodes": bc_eps,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN, "world_dim": fan.RAW_VIEW_DIM,
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
        f"  D3: shaped={hl['d3_rawview_shaped_forage']} bc={hl['d3_rawview_bc_forage']} "
        f"(bc_acc={hl['d3_rawview_bc_action_match_accuracy']}) "
        f"local_view={hl['d3_local_view_greedy_denominator']} "
        f"(any_supra={hl['d3_any_majority_supra_floor']})", flush=True,
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
