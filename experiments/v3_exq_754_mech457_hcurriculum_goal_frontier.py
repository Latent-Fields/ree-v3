#!/opt/local/bin/python3
"""V3-EXQ-754 -- MECH-457 GOV-FANOUT-1 leg H-curriculum (teaching-signal axis) DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457","INV-088"]
tag relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending; INV-088 stays candidate / pending_substrate_reconfirmation.

WHY. V3-EXQ-751 settled the algorithm axis: a learned unsupervised novelty signal (RND) clears
the 1.0 forage floor with NO expert (5.22) but PLATEAUS far below the BC expert (32.72) and the
local_view observability ceiling (48.05). The class-choice /lit-pull
(targeted_review_action_learning_bootstrap_class_choice/SYNTHESIS.md) REJECTED building more
novelty (coverage not competence; duplicates ARC-065/MECH-314) and named FOUR composable
candidate classes. This leg tests the teaching-signal axis: an IMGEP/AMIGo-lite learning-progress
curriculum over goal DISTANCE (Forestier 2022 / Campero 2021) trains a goal-conditioned
actor-critic to reach proposed cells with NO expert; the forage eval selects the goal
UNSUPERVISED from the agent's own 5x5 resource_field_view (nearest visible resource). It is the
UNSUPERVISED analogue of the 748 BC arm -- self-generated frontier instead of expert
demonstrations -- so it directly tests whether a self-generated teaching signal reaches the BC
competence band.

THE MECHANISM (teaching-signal axis; NOT a power-bump of the sparse-RL/novelty design). No
intrinsic novelty bonus is added to foraging; instead a curriculum PROPOSES navigation goals at
an adapting Manhattan distance (raise on high reach-rate, lower on low reach-rate = learning
progress), training a goal-conditioned policy to reach them. The forage eval then EXPLOITS that
navigation competence by picking the goal from OBSERVATION (the 5x5 view), never from privileged
env state. Implemented in experiments/_lib/mech457_explorer_classes.{GoalCondModel,
train_goal_curriculum, GoalCondForagePolicy} -> the arm is substrate-hashed and REUSE-ELIGIBLE.

DECLARED NULL (so a plateaued/sub-floor leg is informative, not wasted): an unsupervised
learning-progress goal curriculum does NOT approach the BC 32.72 competence band
unsupervised -- the self-generated frontier does not substitute for expert demonstrations (route
to the next class or a composition). A NULL here weights AGAINST nothing (diagnostic); it narrows
the build.

REPRESENTATIONS. Run on BOTH z_world(cotrain) AND the raw 5x5 view as SEPARATE reuse-eligible
arms (user directive 2026-07-14): the INV-088 V3-EXQ-750 retest needs matched-competent
unsupervised policies on both representations, and separate arms maximise future reuse. A
matched-competent unsupervised CURRICULUM policy on both reps feeds that representation->diversity
retest -- hence this leg ALSO tags INV-088 relevance.

REFERENCE. The sparse baseline runs SAME-RUN on both reps (OFF/floor reference + reuse mint +
substrate-drift guard). The RND plateau (5.22), BC (32.72), and ceiling (48.05) are CITED
constants (same substrate, days apart -- the 751 precedent cited BC without re-running it).
Anchors (local_view_greedy / greedy_oracle / random_walk) evaluated live for readiness +
denominators.

READINESS (P0 readiness-assert; SAME statistic as the verdict = foraging_competence @D3 vs the
1.0 floor). local_view_greedy (5x5 view) and greedy_oracle clear the floor @D3. Below readiness
-> substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict label).

ethics_preflight: involves_negative_valence=false, involves_suffering_like_state=false,
  involves_self_model=false, involves_inescapability_or_helplessness=false,
  involves_offline_replay_over_harm=false, involves_social_mind_or_language=false,
  involves_human_data_or_clinical_context=false, decision=allow. (V3 foraging RL; SENT-0.)

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech457_explorer_classes.py (mech, imported) +
experiments/_lib/mech457_fanout.py (fan). ASCII-only in all runtime strings.
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
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_754_mech457_hcurriculum_goal_frontier"
QUEUE_ID = "V3-EXQ-754"
CLAIM_IDS: List[str] = ["MECH-457", "INV-088"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

MECH_KEY = "hcurriculum"
# (representation, arm_kind) treatment + reference cells; anchors handled separately.
CELL_ARMS: Tuple[Tuple[str, str], ...] = (
    ("z_world", "sparse"), ("z_world", "hcurriculum"),
    ("raw_view", "sparse"), ("raw_view", "hcurriculum"),
)


def _arm_id(rep: str, kind: str) -> str:
    return f"{kind}_{'zw' if rep == 'z_world' else 'raw'}"


def _arm_config_slice(rep: str, kind: str, env_kwargs: Dict[str, Any], p0: int, rl_eps: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": _arm_id(rep, kind), "mechanism_class": MECH_KEY,
        "representation": rep, "arm_kind": kind, "rung_id": fan.RUNG_ID,
        "env_kwargs": dict(env_kwargs), "eval_episodes": int(eval_eps),
        "steps_per_episode": int(steps), "rl_episodes": int(rl_eps),
        "actor_critic_hidden": mech.ACTOR_CRITIC_HIDDEN, "ac_lr": mech.AC_LR,
        "ac_gamma": mech.AC_GAMMA,
    }
    if rep == "z_world":
        base.update({"kind": "zworld_actor_critic", "cotrain_encoder": True,
                     "use_sf_critic": False, "p0_warmup_episodes": int(p0)})
    else:
        base.update({"kind": "rawview_actor_critic", "raw_view_dim": mech.RAW_VIEW_DIM})
    if kind == "sparse":
        base.update({"teacher": "sparse_foraging_rl", "entropy_beta": mech.AC_ENTROPY_BETA,
                     "intrinsic": "none", "credit_replay": False})
    else:  # hcurriculum
        base.update({"teacher": "unsupervised_goal_conditioned_learning_progress_curriculum",
                     "entropy_beta": mech.EXPLORE_ENTROPY_BETA, "intrinsic": "none",
                     "goal_dim": mech.GOAL_DIM, "curriculum_d0": mech.CURRICULUM_D0,
                     "curriculum_d_max": mech.CURRICULUM_D_MAX,
                     "goal_reach_reward": mech.GOAL_REACH_REWARD})
    return base


def _run_cell(rep: str, kind: str, env_kwargs: Dict[str, Any], seed: int, p0: int, rl_eps: int,
              eval_eps: int, steps: int) -> Dict[str, Any]:
    warm_env = x734._make_env(seed, env_kwargs)
    rep_agent = mech.make_rep(rep, warm_env, seed=seed, p0=p0, steps=steps)
    arm_label = _arm_id(rep, kind)

    train_env = x734._make_env(seed, env_kwargs)
    eval_env = x734._make_env(seed, env_kwargs)

    if kind == "sparse":
        guard = mech.train_a2c(
            rep_agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_label, denom=rl_eps,
        )
        row = evaluate_seed(rep_agent.eval_policy(arm_label), eval_env, eval_eps, steps)
        row["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    else:  # hcurriculum
        model = mech.GoalCondModel(rep, rep_agent)
        guard = mech.train_goal_curriculum(
            model, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_label, denom=rl_eps,
        )
        eval_policy = mech.GoalCondForagePolicy(model, arm_label)
        row = evaluate_seed(eval_policy, eval_env, eval_eps, steps)
        row["final_curriculum_distance"] = guard["final_curriculum_distance"]
        row["final_goal_reach_rate"] = guard["final_goal_reach_rate"]
    return row


def run_experiment(seeds: List[int], p0: int, rl_eps: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg H-curriculum (teaching-signal axis) diagnostic "
        f"({len(CELL_ARMS)} trained arms x 2 reps + {len(fan.ANCHOR_ARMS)} anchors, "
        f"rung {fan.RUNG_ID}, {len(seeds)} seeds; P0={p0} RL={rl_eps} eval={eval_eps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    arm_ids = [_arm_id(r, k) for (r, k) in CELL_ARMS] + list(fan.ANCHOR_ARMS)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in arm_ids}
    per_arm_trainforage: Dict[str, List[float]] = {
        _arm_id(r, k): [] for (r, k) in CELL_ARMS if k == "sparse"
    }
    per_arm_curr_dist: Dict[str, List[float]] = {}
    per_arm_curr_reach: Dict[str, List[float]] = {}
    all_cells: List[Dict[str, Any]] = []

    def _record(arm_id: str, row: Dict[str, Any]) -> None:
        row["rung_id"] = fan.RUNG_ID
        row["arm_id"] = arm_id
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id in per_arm_trainforage:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        if "final_curriculum_distance" in row:
            per_arm_curr_dist.setdefault(arm_id, []).append(float(row["final_curriculum_distance"]))
        if "final_goal_reach_rate" in row:
            per_arm_curr_reach.setdefault(arm_id, []).append(float(row["final_goal_reach_rate"]))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} forage/ep={forage})", flush=True,
        )

    # Anchors first (readiness gate + denominators; rep-agnostic reference policies).
    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
            with arm_cell(seed, config_slice={"arm_id": arm_id, "kind": "anchor",
                                              "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs)},
                          script_path=Path(__file__), config_slice_declared=True,
                          include_driver_script_in_hash=False) as cell:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
                row["seed"] = int(seed)
                cell.stamp(row)
            _record(arm_id, row)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _mean_list(vals: List[float]) -> float:
        return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    readiness_met = mech.readiness_from_anchors(local_view_mean, oracle_mean)

    if readiness_met:
        for (rep, kind) in CELL_ARMS:
            arm_id = _arm_id(rep, kind)
            slice_cfg = _arm_config_slice(rep, kind, env_kwargs, p0, rl_eps, eval_eps, steps)
            for seed in seeds:
                print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
                with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                              config_slice_declared=True,
                              include_driver_script_in_hash=False) as cell:
                    row = _run_cell(rep, kind, env_kwargs, seed, p0, rl_eps, eval_eps, steps)
                    row["seed"] = int(seed)
                    cell.stamp(row)
                _record(arm_id, row)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping trained arms -> substrate_not_ready_requeue", flush=True,
        )

    # ---- Self-route (single-mechanism: does H-curriculum lift above the novelty plateau?) ----
    def _summ(rep: str, kind: str) -> Dict[str, Any]:
        return fan.summarize(per_arm_forage[_arm_id(rep, kind)])

    hcur_zw = _summ("z_world", "hcurriculum")
    hcur_raw = _summ("raw_view", "hcurriculum")
    sp_zw = _summ("z_world", "sparse")
    sp_raw = _summ("raw_view", "sparse")

    def _lifts(summary: Dict[str, Any]) -> bool:
        return bool(summary["majority_supra_floor"]
                    and summary["foraging_competence_mean"] > mech.LIFT_ABOVE_PLATEAU)

    zw_lifts, raw_lifts = _lifts(hcur_zw), _lifts(hcur_raw)
    any_lifts = bool(zw_lifts or raw_lifts)
    any_clears = bool(hcur_zw["majority_supra_floor"] or hcur_raw["majority_supra_floor"])

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_lifts:
        outcome, label = "PASS", "hcurriculum_lifts_unsupervised_forage_above_novelty_plateau"
    elif any_clears:
        outcome, label = "FAIL", "hcurriculum_clears_floor_but_plateaus_at_novelty_band"
    else:
        outcome, label = "FAIL", "hcurriculum_subfloor_does_not_clear_floor"

    degeneracy = check_degeneracy({
        "hcurriculum_vs_sparse_vs_anchor_foraging": {
            "values": [hcur_zw["foraging_competence_mean"], hcur_raw["foraging_competence_mean"],
                       sp_zw["foraging_competence_mean"], sp_raw["foraging_competence_mean"],
                       local_view_mean, _mean("random_walk")]
        }
    })

    interpretation = {
        "label": label,
        "preconditions": [
            fan.readiness_precondition(local_view_mean),
            {"name": "greedy_oracle_clears_floor_at_d3", "kind": "readiness",
             "description": "Env floor-achievable with global info (achievability anchor).",
             "control": "greedy_oracle foraging_competence @D3 vs the 1.0 floor",
             "measured": round(oracle_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
        ],
        "criteria": [
            {"name": "C_hcurriculum_lifts_above_novelty_plateau", "load_bearing": True,
             "passed": bool(any_lifts)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "hcurriculum_vs_sparse_vs_anchor_spread": bool(degeneracy["non_degenerate"]),
        },
    }

    def _tf(arm: str) -> float:
        vals = per_arm_trainforage.get(arm, [])
        return round(float(sum(vals) / len(vals)), 6) if vals else 0.0

    result: Dict[str, Any] = {
        "outcome": outcome,
        "interpretation": interpretation,
        "interpretation_label": label,
        "discrimination_verdict": label,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"MECH-457": "unknown", "INV-088": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
        },
        "headline": {
            "d3_hcurriculum_zworld_forage": hcur_zw["foraging_competence_mean"],
            "d3_hcurriculum_zworld_per_seed": hcur_zw["foraging_competence_per_seed"],
            "d3_hcurriculum_raw_forage": hcur_raw["foraging_competence_mean"],
            "d3_hcurriculum_raw_per_seed": hcur_raw["foraging_competence_per_seed"],
            "d3_sparse_zworld_forage": sp_zw["foraging_competence_mean"],
            "d3_sparse_raw_forage": sp_raw["foraging_competence_mean"],
            "hcurriculum_zworld_lifts_above_plateau": zw_lifts,
            "hcurriculum_raw_lifts_above_plateau": raw_lifts,
            "any_rep_lifts_above_plateau": any_lifts,
            "curriculum_final_distance_zworld": _mean_list(
                per_arm_curr_dist.get(_arm_id("z_world", "hcurriculum"), [])
            ),
            "goal_reach_rate_zworld": _mean_list(
                per_arm_curr_reach.get(_arm_id("z_world", "hcurriculum"), [])
            ),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
        },
        "reference_band": mech.reference_band(),
        "baseline_guard": {
            "load_bearing": False,
            "d3_sparse_zworld_forage": sp_zw["foraging_competence_mean"],
            "sparse_zworld_reproduces_742_band": bool(
                mech.REFERENCE_742_SPARSE_LOW <= sp_zw["foraging_competence_mean"]
                <= mech.REFERENCE_742_SPARSE_HIGH
            ),
            "reference_742_sparse_band": [mech.REFERENCE_742_SPARSE_LOW, mech.REFERENCE_742_SPARSE_HIGH],
            "note": ("The sparse baseline is the SAME-RUN OFF reference (742 sparse cell). Expected "
                     "sub-floor ~0.20-0.27. If it unexpectedly lifts, flag substrate drift; do NOT "
                     "self-route on a confounded baseline."),
        },
        "per_arm": {a: fan.summarize(per_arm_forage[a]) for a in arm_ids},
        "per_arm_train_forage_recent": {
            _arm_id(r, k): _tf(_arm_id(r, k)) for (r, k) in CELL_ARMS if k == "sparse"
        },
        "per_arm_curriculum": {
            _arm_id(r, k): {
                "final_curriculum_distance": _mean_list(
                    per_arm_curr_dist.get(_arm_id(r, k), [])
                ),
                "final_goal_reach_rate": _mean_list(
                    per_arm_curr_reach.get(_arm_id(r, k), [])
                ),
            }
            for (r, k) in CELL_ARMS if k == "hcurriculum"
        },
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
        "reference_band": result["reference_band"],
        "baseline_guard": result["baseline_guard"],
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_train_forage_recent": result["per_arm_train_forage_recent"],
        "per_arm_curriculum": result["per_arm_curriculum"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "portfolio": {
            "gov_fanout_1": "MECH-457 combination-aware discrimination portfolio (752-756)",
            "leg": "H-curriculum (teaching-signal axis)",
            "mechanism": ("IMGEP/AMIGo-lite (Forestier 2022 / Campero 2021): a learning-progress "
                          "curriculum over goal DISTANCE trains a goal-conditioned actor-critic to "
                          "reach proposed cells with NO expert; the forage eval picks the goal "
                          "UNSUPERVISED from the agent's own 5x5 resource_field_view (nearest "
                          "visible resource). The unsupervised analogue of the 748 BC arm."),
            "siblings": ["V3-EXQ-752 (H-credit)", "V3-EXQ-753 (H-return)",
                         "V3-EXQ-755 (H-mode)", "V3-EXQ-756 (H-credit x H-return pair)"],
            "reference_constants": {"rnd_novelty_plateau_751": mech.RND_PLATEAU_5_22,
                                    "bc_expert_748": mech.BC_REFERENCE_32_72,
                                    "local_view_ceiling_738": mech.CEILING_48_05},
        },
        "reuse_mint": {
            "reusable_arms": [_arm_id(r, k) for (r, k) in CELL_ARMS],
            "reuse_eligible": True,
            "note": ("All trained arms (sparse + hcurriculum, both reps) are minted reuse-eligible "
                     "(mint-as-you-go): rng_fully_reset via arm_cell + config_slice_declared + "
                     "include_driver_script_in_hash=False; the mechanism lives in "
                     "experiments/_lib/mech457_explorer_classes.py (in the substrate hash), so "
                     "the cells are anchored on the substrate + config + seed + machine_class, "
                     "NOT this driver. A future MECH-457 consumer (incl. the INV-088 750 retest, "
                     "which needs a matched-competent unsupervised curriculum policy on BOTH reps) "
                     "can cite reuse_baseline_from this run_id."),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 foraging_competence (unshaped eval) of the H-curriculum goal-conditioned arm on "
            "z_world(cotrain) AND raw 5x5 view vs the 1.0 floor AND vs the RND novelty plateau "
            f"(lift threshold {round(mech.LIFT_ABOVE_PLATEAU, 4)}); strict majority of seeds. "
            "Readiness = local_view_greedy + greedy_oracle clear the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg H-curriculum (teaching-signal axis). DIAGNOSTIC (excluded from "
            "scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy before any governance "
            "action. The UNSUPERVISED analogue of the 748 BC arm: a self-generated "
            "learning-progress goal curriculum instead of expert demonstrations. NOT a power-bump "
            "of the sparse-RL/novelty design -- a different design axis (teaching signal), new "
            "mechanism = new EXQ, not a lettered iteration. NULL = the self-generated frontier "
            "does NOT approach the BC 32.72 band unsupervised -> narrows the build to the next "
            "class / a composition. ALSO tags INV-088: a matched-competent unsupervised curriculum "
            "policy on both reps unblocks the V3-EXQ-750 representation->diversity retest. "
            "GOV-REUSE-1: the decisive readout (unsupervised goal-curriculum foraging on both "
            "reps) is recorded in NO prior manifest -> run. Re-derive brake: does NOT fire (0 "
            "substrate_ceiling/non_contributory autopsies on MECH-457; INV-088 count after this "
            "run < threshold). MECH-457 stays candidate/v3_pending; INV-088 stays "
            "candidate/pending_substrate_reconfirmation. Both reps run as separate reuse-eligible "
            "arms for the INV-088 750 retest."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-754 MECH-457 H-curriculum unsupervised goal-frontier diagnostic"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0, rl_eps = fan.DRY_P0, fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0, rl_eps = fan.P0_WARMUP_EPISODES, fan.RL_EPISODES
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, rl_eps=rl_eps, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": [_arm_id(r, k) for (r, k) in CELL_ARMS],
        "anchors": list(fan.ANCHOR_ARMS), "p0_warmup_episodes": p0, "rl_episodes": rl_eps,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": mech.ACTOR_CRITIC_HIDDEN, "cotrain_encoder": True,
        "ac_lr": mech.AC_LR, "ac_gamma": mech.AC_GAMMA,
        "sparse_entropy_beta": mech.AC_ENTROPY_BETA,
        "curriculum_entropy_beta": mech.EXPLORE_ENTROPY_BETA,
        "goal_dim": mech.GOAL_DIM, "curriculum_d0": mech.CURRICULUM_D0,
        "curriculum_d_max": mech.CURRICULUM_D_MAX, "goal_reach_reward": mech.GOAL_REACH_REWARD,
        "lift_above_plateau": mech.LIFT_ABOVE_PLATEAU,
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
        f"  D3 hcurriculum: zw={hl['d3_hcurriculum_zworld_forage']} raw={hl['d3_hcurriculum_raw_forage']} "
        f"sparse_zw={hl['d3_sparse_zworld_forage']} local_view={hl['d3_local_view_greedy_denominator']} "
        f"(any_lifts={hl['any_rep_lifts_above_plateau']})", flush=True,
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
