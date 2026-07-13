#!/opt/local/bin/python3
"""V3-EXQ-747 -- MECH-457 GOV-FANOUT-1 leg A (H-rep) -- raw-5x5-view actor-critic DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending.

WHY. V3-EXQ-742 refuted the "single missing mechanism = MECH-457" reading: all four first-class
RPE actor-critic arms forage 0.20-0.27/ep at D3 (below random_walk 0.93; ~0.5% of the
local_view_greedy ceiling 48.05) with every readiness precondition met. The 742 autopsy
(failure_autopsy_morning-digest-742-744a-745-746-746a_2026-07-13) routed the surviving deficit
-- UPSTREAM of the actor-critic head -- to a GOV-FANOUT-1 2x2 discrimination portfolio and
REFUSED a same-question floor re-pose. This leg is the REPRESENTATION axis.

THE 2x2 FACTORIAL (742 = the (z_world, sparse) cell):
                     sparse foraging RL        dense teacher (shaping + BC)
    z_world  (R0)    742: FAIL (cited)         V3-EXQ-748 (H-explore)
    raw 5x5  (R1)    THIS (V3-EXQ-747)         V3-EXQ-749 (conjunction)

H-rep HYPOTHESIS. The prediction-trained z_world is action-inadequate: the encoder->policy
interface is the wall. TEST: train a STANDALONE ActorCriticPolicy(world_dim=25) directly on the
raw 5x5 resource_field_view -- the EXACT input LocalViewGreedyPolicy uses (48.05 @D3) --
bypassing z_world entirely, with the SAME sparse foraging teacher as 742. No REE encoder, no P0
warmup.

DECLARED NULL (so a sub-floor leg is informative, not wasted):
  * treatment clears the 1.0 floor -> the raw view IS action-learnable from sparse reward while
    z_world (742) is not -> z_world is the wall (build an action-adequate encoder/observation
    path). SELF-ROUTE: rawview_learns_zworld_is_the_wall.
  * treatment sub-floor -> the raw view + sparse RL still cannot be learned -> representation is
    NOT the sole wall (does NOT mean representation is irrelevant; the conjunction leg 749 tests
    raw-view + a DENSE teacher). SELF-ROUTE: rawview_sparse_insufficient (points to the reward
    axis / interaction; NOT a substrate ceiling -- readiness proves the env is solvable).

READINESS (P0 readiness-assert; same statistic as the verdict). The load-bearing criterion
reads a LEARNED quantity (the trained actor's foraging_competence @D3 vs the 1.0 floor), so the
readiness precondition asserts the SAME statistic on a known-positive control:
LocalViewGreedyPolicy reading the SAME 5x5 view forages >= floor @D3 (plus greedy_oracle >=
floor for achievability). Below readiness -> substrate_not_ready_requeue (FAIL; NEVER a
substrate-verdict label) and the treatment training is SKIPPED (assert before the expensive
measurement).

evidence_direction = "unknown" (a DIAGNOSTIC that discriminates WHERE the wall is; it does not
directly score MECH-457 -- the discrimination verdict lives in interpretation.label /
discrimination_verdict, adjudicated by /failure-autopsy).

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

Shared machinery: experiments/_lib/mech457_fanout.py (raw-view AC trainer, anchors, readiness,
self-route scaffolding). ASCII-only in all runtime strings.
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

EXPERIMENT_TYPE = "v3_exq_747_mech457_hrep_rawview_actor_critic"
QUEUE_ID = "V3-EXQ-747"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

TREATMENT_ARM = "ac_rawview_sparse_rl"
ARM_ORDER: Tuple[str, ...] = (TREATMENT_ARM,) + fan.ANCHOR_ARMS


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], rl_eps: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    if arm_id == TREATMENT_ARM:
        base.update({
            "kind": "rawview_actor_critic", "representation": "raw_5x5_resource_field_view",
            "teacher": "sparse_foraging", "shaping_coef": 0.0,
            "rl_episodes": int(rl_eps), "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
            "world_dim": fan.RAW_VIEW_DIM,
        })
    else:
        base.update({"kind": "anchor"})
    return base


def _run_treatment_cell(env_kwargs: Dict[str, Any], seed: int, rl_eps: int,
                        eval_eps: int, steps: int) -> Dict[str, Any]:
    ac = fan.make_rawview_ac()
    train_env = x734._make_env(seed, env_kwargs)
    guard = fan.train_rawview_ac_rl(
        ac, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
        arm_label=TREATMENT_ARM, denom=rl_eps, shaping_coef=0.0,
    )
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(fan.RawViewACEvalPolicy(ac, TREATMENT_ARM), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    return row


def run_experiment(seeds: List[int], rl_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg A (H-rep raw-view) diagnostic "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"RL={rl_eps}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_trainforage: Dict[str, List[float]] = {TREATMENT_ARM: []}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, rl_eps, eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id == TREATMENT_ARM:
                row = _run_treatment_cell(env_kwargs, seed, rl_eps, eval_eps, steps)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id == TREATMENT_ARM:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    # --- anchors FIRST (readiness assert before the expensive treatment training) ----------
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

    # --- treatment (only if the env is ready) ----------------------------------------------
    if readiness_met:
        for seed in seeds:
            _run_cell(TREATMENT_ARM, seed)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping treatment training -> substrate_not_ready_requeue", flush=True,
        )

    treatment = fan.summarize(per_arm_forage[TREATMENT_ARM])
    treatment_maj = bool(treatment["majority_supra_floor"])

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif treatment_maj:
        outcome, label = "PASS", "rawview_learns_zworld_is_the_wall"
    else:
        outcome, label = "FAIL", "rawview_sparse_insufficient"

    # --- non-degeneracy (yardstick spread across the compared arms) ------------------------
    degeneracy = check_degeneracy({
        "d3_rawview_vs_anchor_foraging": {
            "values": [treatment["foraging_competence_mean"], local_view_mean, _mean("random_walk")]
        }
    })

    # reward-hacking / bootstrap guard (instrument-only): train-vs-eval foraging divergence.
    tf = per_arm_trainforage[TREATMENT_ARM]
    train_mean = round(float(sum(tf) / len(tf)), 6) if tf else 0.0
    eval_mean = treatment["foraging_competence_mean"]

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
            {"name": "C_rawview_actor_clears_floor_at_D3", "load_bearing": True,
             "passed": bool(treatment_maj)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "rawview_vs_anchor_foraging_spread": bool(degeneracy["non_degenerate"]),
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
            "d3_rawview_sparse_forage": treatment["foraging_competence_mean"],
            "d3_rawview_sparse_per_seed": treatment["foraging_competence_per_seed"],
            "d3_rawview_majority_supra_floor": treatment_maj,
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "rawview_normalized_frac_of_local_view_d3": (
                round(treatment["foraging_competence_mean"] / local_view_mean, 6)
                if local_view_mean > 1e-9 else None
            ),
            "reference_742_zworld_sparse_forage_d3": "0.20-0.27 (cited, not re-run)",
        },
        "bootstrap_guard": {
            "load_bearing": False,
            "d3_mean_train_forage_recent": train_mean,
            "d3_eval_foraging": round(eval_mean, 6),
            "train_vs_eval_divergence": round(eval_mean - train_mean, 6),
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
            "leg": "A (H-rep, representation axis)",
            "factorial_cell": "(raw_5x5_view, sparse_foraging_RL)",
            "siblings": ["V3-EXQ-748 (H-explore)", "V3-EXQ-749 (conjunction)"],
            "reference_cell_742": "(z_world, sparse) -> FAIL (cited, not re-run)",
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 raw-5x5-view actor-critic foraging_competence (mean resources/ep, unshaped) vs "
            "the 1.0 floor, strict majority of seeds; readiness = local_view_greedy (same view) "
            "clears the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg A. DIAGNOSTIC (excluded from scoring); PROMOTES/DEMOTES NOTHING; "
            "route to /failure-autopsy before any governance action. Same-question 742 floor "
            "re-pose REFUSED -- this attacks the REPRESENTATION axis (raw view bypassing "
            "z_world), a different mechanism. MECH-457 stays candidate/v3_pending. GOV-REUSE-1: "
            "decisive readout (actor-critic foraging on the raw view) is recorded in NO prior "
            "manifest (742 was z_world+sparse only) -> run."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-747 MECH-457 H-rep raw-view actor-critic diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        rl_eps, eval_eps, steps = fan.DRY_RL, fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        rl_eps, eval_eps, steps = fan.RL_EPISODES, fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, rl_eps=rl_eps, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "rl_episodes": rl_eps, "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN, "world_dim": fan.RAW_VIEW_DIM,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "teacher": "sparse_foraging (FORAGE_BONUS per resource + harm + novelty)",
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
        f"  D3: rawview_sparse={hl['d3_rawview_sparse_forage']} "
        f"local_view={hl['d3_local_view_greedy_denominator']} "
        f"(majority_supra={hl['d3_rawview_majority_supra_floor']})", flush=True,
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
