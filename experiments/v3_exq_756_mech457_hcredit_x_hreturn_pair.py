#!/opt/local/bin/python3
"""V3-EXQ-756 -- MECH-457 GOV-FANOUT-1 leg H-credit x H-return PAIR (combination cell) DIAGNOSTIC.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457
stays candidate / v3_pending.

WHY. The class-choice /lit-pull named FOUR composable candidate classes closing the RND
floor->competent gap (5.22 novelty plateau, far below BC 32.72 and the 48.05 local_view
ceiling). The user directive 2026-07-14 is combination-aware: the gap has TWO plausibly-additive
halves -- a credit-PROPAGATION sub-problem (H-credit) and a frontier-RETURN / detachment
sub-problem (H-return). Each alone may under-close; together they may be COMPLEMENTARY. This leg
runs the PAIR (both top-composition classes together) ALONGSIDE each alone in the SAME RUN so
additivity/complementarity is directly readable off matched seeds -- no cross-run comparability
risk.

THE MECHANISM (combination cell; NOT a power-bump of either single design). The pair arm applies
BOTH top-composition classes together:
  * H-credit  -- prioritized backward credit replay (|TD-error|-prioritized Mattar&Daw 2018
        reverse-replay Foster&Wilson 2006 extra AC updates on reward-bearing trajectories; the
        RL-native homolog of hippocampal backward_credit_sweep MECH-290). ONLY the update
        changes; rollout/exploration is byte-identical to sparse.
  * H-return  -- a Go-Explore (Ecoffet 2021) archive of restorable frontier snapshots; at
        episode start, with return_prob, env.reset_to() a selected frontier cell and explore
        on-policy from there (the derailment/detachment fix).
Both single classes (H-credit alone, H-return alone) and the sparse baseline run SAME-RUN so the
pair's gain over the BETTER single is read on matched seeds. All hooks live in
experiments/_lib/mech457_explorer_classes.train_a2c(credit_replay / archive / return_prob) -> the
arms are substrate-hashed and REUSE-ELIGIBLE.

DECLARED NULL (so a non-additive/sub-floor leg is informative, not wasted): the pair does NOT
exceed the BETTER single by >= mech.ADDITIVITY_ABS_MARGIN -> credit-propagation and
frontier-return are NOT complementary for closing the floor->competent gap (the two halves are
redundant, or one dominates). A NULL here weights AGAINST nothing (diagnostic); it narrows the
build (stop pairing these two; try a different axis).

REPRESENTATIONS. Run on BOTH z_world(cotrain) AND the raw 5x5 view as SEPARATE reuse-eligible
arms (user directive 2026-07-14): separate arms maximise future reuse and the INV-088 750 retest
needs matched-competent unsupervised policies on both representations.

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

EXPERIMENT_TYPE = "v3_exq_756_mech457_hcredit_x_hreturn_pair"
QUEUE_ID = "V3-EXQ-756"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

MECH_KEY = "pair"
# (representation, arm_kind) treatments + reference cells; anchors handled separately.
# FOUR arm kinds per rep: sparse (OFF/floor) + the two top-composition singles + their pair.
CELL_ARMS: Tuple[Tuple[str, str], ...] = (
    ("z_world", "sparse"), ("z_world", "hcredit"), ("z_world", "hreturn"), ("z_world", "pair"),
    ("raw_view", "sparse"), ("raw_view", "hcredit"), ("raw_view", "hreturn"), ("raw_view", "pair"),
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
    elif kind == "hcredit":
        base.update({"teacher": "sparse_rl_plus_prioritized_backward_credit_replay",
                     "entropy_beta": mech.AC_ENTROPY_BETA, "intrinsic": "none",
                     "credit_replay": True, "credit_replay_passes": mech.CREDIT_REPLAY_PASSES,
                     "credit_topk": mech.CREDIT_TOPK, "credit_gamma": mech.CREDIT_GAMMA})
    elif kind == "hreturn":
        base.update({"teacher": "sparse_rl_plus_go_explore_archive_return",
                     "entropy_beta": mech.AC_ENTROPY_BETA, "intrinsic": "none",
                     "credit_replay": False, "return_prob": mech.RETURN_PROB,
                     "archive_cell_size": mech.ARCHIVE_CELL_SIZE, "archive_max": mech.ARCHIVE_MAX})
    else:  # pair
        base.update({"teacher": "sparse_rl_plus_credit_replay_and_go_explore_return",
                     "entropy_beta": mech.AC_ENTROPY_BETA, "intrinsic": "none",
                     "credit_replay": True, "return_prob": mech.RETURN_PROB,
                     "credit_replay_passes": mech.CREDIT_REPLAY_PASSES,
                     "credit_topk": mech.CREDIT_TOPK, "credit_gamma": mech.CREDIT_GAMMA,
                     "archive_cell_size": mech.ARCHIVE_CELL_SIZE, "archive_max": mech.ARCHIVE_MAX})
    return base


def _run_cell(rep: str, kind: str, env_kwargs: Dict[str, Any], seed: int, p0: int, rl_eps: int,
              eval_eps: int, steps: int) -> Dict[str, Any]:
    warm_env = x734._make_env(seed, env_kwargs)
    rep_agent = mech.make_rep(rep, warm_env, seed=seed, p0=p0, steps=steps)
    arm_label = _arm_id(rep, kind)

    train_env = x734._make_env(seed, env_kwargs)
    # Mechanism hooks per arm kind. GoExploreArchive constructed HERE (inside the arm_cell
    # RNG-reset scope) so the archive RNG is seeded within the fingerprinted cell.
    hooks: Dict[str, Any] = {}
    if kind == "hcredit":
        hooks["credit_replay"] = True
    elif kind == "hreturn":
        hooks["archive"] = mech.GoExploreArchive(seed)
        hooks["return_prob"] = mech.RETURN_PROB
    elif kind == "pair":
        hooks["credit_replay"] = True
        hooks["archive"] = mech.GoExploreArchive(seed)
        hooks["return_prob"] = mech.RETURN_PROB

    guard = mech.train_a2c(
        rep_agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
        arm_label=arm_label, denom=rl_eps, **hooks,
    )

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_label), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = guard["mean_train_forage_recent"]
    row["n_credit_replay_passes"] = guard["n_credit_replay_passes"]
    row["n_return_episodes"] = guard["n_return_episodes"]
    return row


def run_experiment(seeds: List[int], p0: int, rl_eps: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 leg H-credit x H-return PAIR (combination cell) diagnostic "
        f"({len(CELL_ARMS)} trained arms x 2 reps + {len(fan.ANCHOR_ARMS)} anchors, "
        f"rung {fan.RUNG_ID}, {len(seeds)} seeds; P0={p0} RL={rl_eps} eval={eval_eps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    arm_ids = [_arm_id(r, k) for (r, k) in CELL_ARMS] + list(fan.ANCHOR_ARMS)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in arm_ids}
    per_arm_trainforage: Dict[str, List[float]] = {_arm_id(r, k): [] for (r, k) in CELL_ARMS}
    all_cells: List[Dict[str, Any]] = []

    def _record(arm_id: str, row: Dict[str, Any]) -> None:
        row["rung_id"] = fan.RUNG_ID
        row["arm_id"] = arm_id
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id in per_arm_trainforage:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
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

    # ---- Self-route (PAIR/ADDITIVITY: does the pair EXCEED the better single by a margin?) ----
    def _summ(rep: str, kind: str) -> Dict[str, Any]:
        return fan.summarize(per_arm_forage[_arm_id(rep, kind)])

    pair_zw = _summ("z_world", "pair")
    credit_zw = _summ("z_world", "hcredit")
    ret_zw = _summ("z_world", "hreturn")
    pair_raw = _summ("raw_view", "pair")
    credit_raw = _summ("raw_view", "hcredit")
    ret_raw = _summ("raw_view", "hreturn")
    sp_zw = _summ("z_world", "sparse")
    sp_raw = _summ("raw_view", "sparse")

    def _pair_additive(pair_s: Dict[str, Any], credit_s: Dict[str, Any],
                       ret_s: Dict[str, Any]) -> bool:
        best_single = max(credit_s["foraging_competence_mean"], ret_s["foraging_competence_mean"])
        return bool(pair_s["majority_supra_floor"]
                    and pair_s["foraging_competence_mean"] > best_single + mech.ADDITIVITY_ABS_MARGIN)

    zw_add = _pair_additive(pair_zw, credit_zw, ret_zw)
    raw_add = _pair_additive(pair_raw, credit_raw, ret_raw)
    any_add = bool(zw_add or raw_add)
    any_clears = bool(
        credit_zw["majority_supra_floor"] or credit_raw["majority_supra_floor"]
        or ret_zw["majority_supra_floor"] or ret_raw["majority_supra_floor"]
        or pair_zw["majority_supra_floor"] or pair_raw["majority_supra_floor"]
    )

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_add:
        outcome, label = "PASS", "pair_credit_x_return_additive_complementary"
    elif any_clears:
        outcome, label = "FAIL", "pair_no_additivity_over_best_single"
    else:
        outcome, label = "FAIL", "pair_subfloor_does_not_clear_floor"

    pair_gain_zw = pair_zw["foraging_competence_mean"] - max(
        credit_zw["foraging_competence_mean"], ret_zw["foraging_competence_mean"]
    )
    pair_gain_raw = pair_raw["foraging_competence_mean"] - max(
        credit_raw["foraging_competence_mean"], ret_raw["foraging_competence_mean"]
    )

    degeneracy = check_degeneracy({
        "pair_arms_spread": {
            "values": [pair_zw["foraging_competence_mean"], pair_raw["foraging_competence_mean"],
                       credit_zw["foraging_competence_mean"], credit_raw["foraging_competence_mean"],
                       ret_zw["foraging_competence_mean"], ret_raw["foraging_competence_mean"],
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
            {"name": "C_pair_exceeds_best_single_by_margin", "load_bearing": True,
             "passed": bool(any_add)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "pair_arms_spread": bool(degeneracy["non_degenerate"]),
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
        "evidence_direction_per_claim": {"MECH-457": "unknown"},
        "readiness": {
            "readiness_met": readiness_met,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
        },
        "headline": {
            "d3_pair_zworld_forage": pair_zw["foraging_competence_mean"],
            "d3_pair_zworld_per_seed": pair_zw["foraging_competence_per_seed"],
            "d3_pair_raw_forage": pair_raw["foraging_competence_mean"],
            "d3_pair_raw_per_seed": pair_raw["foraging_competence_per_seed"],
            "d3_hcredit_zworld_forage": credit_zw["foraging_competence_mean"],
            "d3_hcredit_zworld_per_seed": credit_zw["foraging_competence_per_seed"],
            "d3_hcredit_raw_forage": credit_raw["foraging_competence_mean"],
            "d3_hcredit_raw_per_seed": credit_raw["foraging_competence_per_seed"],
            "d3_hreturn_zworld_forage": ret_zw["foraging_competence_mean"],
            "d3_hreturn_zworld_per_seed": ret_zw["foraging_competence_per_seed"],
            "d3_hreturn_raw_forage": ret_raw["foraging_competence_mean"],
            "d3_hreturn_raw_per_seed": ret_raw["foraging_competence_per_seed"],
            "d3_sparse_zworld_forage": sp_zw["foraging_competence_mean"],
            "d3_sparse_raw_forage": sp_raw["foraging_competence_mean"],
            "pair_gain_over_best_single_zworld": round(pair_gain_zw, 6),
            "pair_gain_over_best_single_raw": round(pair_gain_raw, 6),
            "pair_additive_zworld": zw_add,
            "pair_additive_raw": raw_add,
            "any_rep_pair_additive": any_add,
            "additivity_abs_margin": round(mech.ADDITIVITY_ABS_MARGIN, 6),
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
        "per_arm_train_forage_recent": {_arm_id(r, k): _tf(_arm_id(r, k)) for (r, k) in CELL_ARMS},
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
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "portfolio": {
            "gov_fanout_1": "MECH-457 combination-aware discrimination portfolio (752-756)",
            "leg": "H-credit x H-return PAIR (combination cell)",
            "mechanism": ("both top-composition classes together: prioritized backward credit "
                          "replay (H-credit; |TD-error|-prioritized Mattar&Daw 2018 reverse-replay "
                          "Foster&Wilson 2006, RL-native homolog of hippocampal "
                          "backward_credit_sweep MECH-290) AND a Go-Explore (Ecoffet 2021) "
                          "archive/return of restorable frontier snapshots (H-return; the "
                          "detachment/derailment fix). Run ALONGSIDE each alone SAME-RUN so "
                          "additivity/complementarity is read off matched seeds."),
            "siblings": ["V3-EXQ-752 (H-credit)", "V3-EXQ-753 (H-return)",
                         "V3-EXQ-754 (H-curriculum)", "V3-EXQ-755 (H-mode)"],
            "reference_constants": {"rnd_novelty_plateau_751": mech.RND_PLATEAU_5_22,
                                    "bc_expert_748": mech.BC_REFERENCE_32_72,
                                    "local_view_ceiling_738": mech.CEILING_48_05},
        },
        "reuse_mint": {
            "reusable_arms": [_arm_id(r, k) for (r, k) in CELL_ARMS],
            "reuse_eligible": True,
            "note": ("All trained arms (sparse + hcredit + hreturn + pair, both reps) are minted "
                     "reuse-eligible (mint-as-you-go): rng_fully_reset via arm_cell + "
                     "config_slice_declared + include_driver_script_in_hash=False; the mechanisms "
                     "live in experiments/_lib/mech457_explorer_classes.py (in the substrate hash), "
                     "so the cells are anchored on the substrate + config + seed + machine_class, "
                     "NOT this driver. A future MECH-457 consumer (incl. the INV-088 750 retest) "
                     "can cite reuse_baseline_from this run_id for any of the 8 cell arms."),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 foraging_competence (unshaped eval) of the PAIR arm on z_world(cotrain) AND raw "
            "5x5 view EXCEEDS the BETTER of the two singles (H-credit / H-return) by >= "
            f"{round(mech.ADDITIVITY_ABS_MARGIN, 4)} resource/ep (complementarity), strict "
            "majority of seeds, and clears the 1.0 floor. Readiness = local_view_greedy + "
            "greedy_oracle clear the floor @D3."
        ),
        "notes": (
            "GOV-FANOUT-1 leg H-credit x H-return PAIR (combination cell). DIAGNOSTIC (excluded "
            "from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy before any "
            "governance action. Combination-aware (user directive 2026-07-14: the gap has two "
            "plausibly-additive halves -- credit-propagation + frontier-return). Runs the pair "
            "ALONGSIDE each single alone SAME-RUN so additivity/complementarity is read on matched "
            "seeds. NOT a power-bump of either single design (new combination = new EXQ, not a "
            "lettered iteration). NULL = the pair does NOT exceed the better single by >= "
            f"{round(mech.ADDITIVITY_ABS_MARGIN, 4)} (no complementarity; the two halves are "
            "redundant or one dominates) -> narrows the build. GOV-REUSE-1: the decisive readout "
            "(foraging under BOTH mechanisms together, matched against each alone) is recorded in "
            "NO prior manifest -> run. Re-derive brake: does NOT fire (0 "
            "substrate_ceiling/non_contributory autopsies on MECH-457; a DIFFERENT combination on "
            "DIFFERENT axes = new EXQ). MECH-457 stays candidate/v3_pending. Both reps run as "
            "separate reuse-eligible arms for the INV-088 750 retest."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-756 MECH-457 H-credit x H-return PAIR combination diagnostic"
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
        "ac_lr": mech.AC_LR, "ac_gamma": mech.AC_GAMMA, "entropy_beta": mech.AC_ENTROPY_BETA,
        "credit_replay_passes": mech.CREDIT_REPLAY_PASSES, "credit_topk": mech.CREDIT_TOPK,
        "credit_gamma": mech.CREDIT_GAMMA, "return_prob": mech.RETURN_PROB,
        "archive_cell_size": mech.ARCHIVE_CELL_SIZE, "archive_max": mech.ARCHIVE_MAX,
        "additivity_abs_margin": mech.ADDITIVITY_ABS_MARGIN,
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
        f"  D3 pair: zw={hl['d3_pair_zworld_forage']} raw={hl['d3_pair_raw_forage']} "
        f"credit_zw={hl['d3_hcredit_zworld_forage']} return_zw={hl['d3_hreturn_zworld_forage']} "
        f"sparse_zw={hl['d3_sparse_zworld_forage']} local_view={hl['d3_local_view_greedy_denominator']} "
        f"(gain_zw={hl['pair_gain_over_best_single_zworld']} any_add={hl['any_rep_pair_additive']})",
        flush=True,
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
