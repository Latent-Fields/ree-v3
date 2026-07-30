#!/opt/local/bin/python3
"""V3-EXQ-748a -- MECH-457 H-rep PRECAUTIONARY RE-DERIVATION under the validated SD-070
zworld_p0-trained encoder -- confirmatory-hardening DIAGNOSTIC (experiment_purpose=diagnostic;
claim_ids=["MECH-457"] tags relevance only -> excluded from governance confidence/conflict
scoring). PROMOTES / DEMOTES NOTHING on its own; routes to /failure-autopsy for adjudication
before any governance action. MECH-457 stays candidate / v3_pending.

DOES NOT SUPERSEDE V3-EXQ-748. 748's PASS (dense_teacher_on_zworld_clears_sparsity_was_the_wall)
is STRUCTURALLY UNCONFOUNDED by the V3-EXQ-780 frozen-random-projection defect: 748 built its
z_world arms with cotrain_encoder=True (an actively co-training mechanism -- the policy gradient
reaches split_encoder.world_encoder through agent.actor_critic_step), NOT the detached
zworld_p0-warmup-then-frozen mechanism V3-EXQ-780 found broken (that defect requires a P0 loop
that DETACHES z_world before any optimizer sees it; 748 has no such detach). See
failure_autopsy_V3-EXQ-819a_2026-07-30.md, the interactive Step 8 gate.

WHY QUEUE THIS ANYWAY. Despite that structural clearance, the frozen-instrument confound has
been central across the whole competence_floor / MECH-457 campaign (sixteen legs ran a z_world
arm on a frozen random projection before V3-EXQ-780 diagnosed it; V3-EXQ-819a then confirmed a
PREDICTION-TRAINED z_world genuinely beats a frozen one on installed foraging competence). Given
that history, the user asked (819a's Step 8 interactive gate, 2026-07-30) for a precautionary
DIRECT re-derivation: re-pose 748's own dense-teacher/z_world-BC question under the now-validated
trained instrument, rather than resting solely on the structural argument that 748 was clean.

THE ONE CHANGE FROM 748 (everything else held fixed for comparability):
  748  (cotrain path):  make_zworld_agent(cotrain=True)  -- encoder co-trained BY the AC/BC
                        gradient itself; starts random, may or may not end up meaningfully
                        prediction-shaped, and is entangled with the policy's own learning.
  748a (this):          make_zworld_agent(cotrain=False) + warmup_zworld(zworld_p0=60) -- the
                        SD-070 P0a recipe PREDICTION-TRAINS the encoder ahead of time (same
                        validated operating point as V3-EXQ-819a: ZWORLD_P0_EPISODES=60), then
                        the representation is DETACHED (frozen) through BC + RL. Encoder quality
                        and downstream policy learning are cleanly separated -- exactly the
                        instrument-validity discipline 819a established, applied here to the
                        SAME dense-teacher/z_world-BC design 748 used.

THE 2x2 FACTORIAL THIS SITS IN (742 = the (z_world, sparse) cell; unchanged from 748):
                     sparse foraging RL        dense teacher (shaping + BC)
    z_world  (R0)    742: FAIL (cited)         748 (cotrain) / 748a (zworld_p0, THIS)
    raw 5x5  (R1)    V3-EXQ-747 (H-rep)        V3-EXQ-749 (conjunction)

HYPOTHESIS UNDER RE-TEST (H-rep, hypothesis_space_registry.v1.json "representation
insufficient" -- eliminated 2026-07-13 on 748's evidence). Same two treatment arms as 748:
  * ac_zworld_shaped_rl -- 742's foraging reward + potential-based distance-to-nearest-resource
    shaping (Ng et al. 1999), trained on the frozen zworld_p0-trained representation.
  * ac_zworld_bc        -- supervised behavior-cloning of LocalViewGreedyPolicy through the
    (frozen, prediction-trained) z_world path. A failed CE fit here is a DIRECT
    "prediction-trained z_world is action-inadequate" signal (recorded as
    bc_action_match_accuracy).

DECLARED NULL, reframed for CONFIRMATORY (not discovery) framing -- this is precautionary
hardening of an already-eliminated hypothesis, not a live discrimination:
  * either dense arm clears the 1.0 floor -> H-rep's 2026-07-13 elimination SURVIVES contact
    with the validated trained instrument. Closes the precaution; nothing changes about H-rep's
    status. SELF-ROUTE: hrep_elimination_confirmed_under_trained_instrument.
  * both dense arms sub-floor (with the encoder CONFIRMED trained -- see readiness below) ->
    a GENUINE, NOTABLE finding: the trained-instrument z_world path cannot support dense-teacher
    installation where the cotrain path (748) could. This would warrant re-opening H-rep for
    real reconsideration, not filing away. SELF-ROUTE: hrep_reopened_under_trained_instrument.

GOV-REUSE-1 (existing-evidence check, before writing this script). Decisive readout: z_world
actor-critic foraging_competence under a DENSE teacher (shaped-RL or BC), with the z_world
encoder PREDICTION-TRAINED via SD-070 zworld_p0=60 and then FROZEN. Queried
scripts/reanalysis_query.py (REE_assembly) for this readout across all recorded manifests,
grouped by substrate_hash: 748/747/749/750/751 share substrate_hash 2eff4545309a504f (the
PRE-SD-070 substrate); 819/819a run on distinct hashes (2b8ac42ee1c06fea / 20cc73239cea3cfc,
POST-SD-070) but test the z_world_random_proj-vs-z_world_trained INSTALL/RETENTION contrast
under a SPARSE teacher (SHAPING_COEF=0.0 in their baseline module), not the dense-teacher
shaped-RL/BC design. NO manifest on any substrate_hash carries "dense-teacher z_world-BC/shaped-
RL under a zworld_p0-trained encoder" -- NOT RECOVERABLE (needs a new manipulation combination
absent from every recorded run) -> proceed to run.

RE-DERIVE BRAKE (Step 2.5b). The corpus scan counts 8 prior substrate_ceiling/non_contributory
autopsies tagging MECH-457 (746c-756; fanout-751-750; fanout-752-753-754; fanout-755;
fanout-770-771-772; gov-fanout-1-cluster-780-781-782; V3-EXQ-765; V3-EXQ-769) -- past the
threshold of 2. None of those eight is a same-axis re-pose of the 747/748/749 dense-teacher
factorial specifically (they discriminate drive-schedule / approach-primitive / retention /
advantage-composition / metabolic-forage / dense-credit / bc-prior axes); V3-EXQ-748 itself
PASSED and was never one of the counted ceiling targets. This queue entry re-tests 748's own
PASS under a corrected instrument, which is exactly the "substrate now BUILT" release condition
in the brake's own text: the upstream substrate the campaign lacked -- SD-070's zworld_p0
prediction-training recipe -- is now IMPLEMENTED (ree-v3 CLAUDE.md, "SD-070 ADOPTION in the
_train_all_on_agent driver family", 2026-07-20; the underlying recipe itself IMPLEMENTED
2026-07-18) and VALIDATED (V3-EXQ-819a, 2026-07-27/2026-07-30: the trained arm confers a
measured competence advantage over a frozen random projection). Brake released for this
specific lineage on that ground.

DV-SYMMETRY INVARIANCE (mandatory declaration, both arms). The manipulation is "prediction-train
the z_world encoder via SD-070 zworld_p0=60, then freeze it" vs 748's "co-train the encoder
inside the AC/BC gradient itself" -- a change to the REPRESENTATION the downstream policy is
built on, not a broadcast additive constant, a monotone rescaling, or a permutation of
interchangeable units. foraging_competence (a trained-policy behavioural outcome under greedy/
argmax eval) is not provably invariant under any symmetry of that manipulation: a different
input representation changes what the BC cross-entropy fit and the RL policy gradient can learn
to do with it. So a measured delta here is a real measurement, not a pre-determined arithmetic
identity, for both ac_zworld_shaped_rl and ac_zworld_bc.

READINESS (P0 readiness-assert; same statistic as the verdict, via the shared
zworld_encoder_guard shaper so this driver adjudicates identically to 728/734/742/819a).
LocalViewGreedyPolicy (5x5 view) and greedy_oracle must clear the 1.0 floor @D3 (env solvable),
AND the P0a zworld_p0 warmup must have moved at least one split_encoder.world_encoder tensor on
the worst treatment cell (encoder genuinely prediction-trained, not a frozen random projection --
GATING per zworld_precondition's own documented policy, since this run's question is directly
ABOUT a learned z_world). Below either -> substrate_not_ready_requeue (FAIL; NEVER a
substrate-verdict label); a below-floor encoder-trained reading is answerable only by re-running,
never by concluding H-rep is reopened.

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

Shared machinery: experiments/_lib/mech457_fanout.py (warmup_zworld's zworld_p0 path, added
2026-07-18/22 for the SD-070 adoption) and experiments/_lib/zworld_encoder_guard.py
(zworld_precondition shaper). ASCII-only in all runtime strings.
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
from experiments._lib.zworld_encoder_guard import zworld_precondition  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_742_mech457_actor_critic_onoff as x742  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_748a_mech457_hrep_zworldp0_rederivation"
QUEUE_ID = "V3-EXQ-748a"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

TREATMENT_ARMS: Tuple[str, ...] = ("ac_zworld_shaped_rl", "ac_zworld_bc")
ARM_ORDER: Tuple[str, ...] = TREATMENT_ARMS + fan.ANCHOR_ARMS

# The ONE change from 748: the encoder is prediction-trained (SD-070 P0a) then frozen, instead
# of co-trained inside the AC/BC gradient. ZWORLD_P0_EPISODES=60 is the SD-070-validated
# operating point (also V3-EXQ-819a's config).
ZWORLD_P0_EPISODES = fan.ZWORLD_P0_EPISODES  # 60
DRY_ZWORLD_P0 = 4


def _arm_config_slice(arm_id: str, env_kwargs: Dict[str, Any], p0: int, zworld_p0: int,
                      rl_eps: int, bc_eps: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
    }
    if arm_id in TREATMENT_ARMS:
        base.update({
            "kind": "zworld_actor_critic", "representation": "z_world_zworld_p0_frozen",
            "cotrain_encoder": False, "zworld_p0_episodes": int(zworld_p0),
            "use_sf_critic": False, "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
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
                        zworld_p0: int, rl_eps: int, bc_eps: int, eval_eps: int, steps: int,
                        dry_run: bool) -> Dict[str, Any]:
    warm_env = x734._make_env(seed, env_kwargs)
    # cotrain=False: z_world DETACHED through BC + RL (819a's instrument-validity discipline),
    # so only the P0a warmup below can shape the encoder -- never the downstream AC/BC gradient.
    agent = fan.make_zworld_agent(warm_env, cotrain=False)
    guard = fan.warmup_zworld(
        agent, warm_env, seed=seed, p0=p0, steps=steps, strict=False,
        zworld_p0=zworld_p0, zworld_p0_dry_run=bool(dry_run),
    )

    train_env = x734._make_env(seed, env_kwargs)
    extra: Dict[str, Any] = {}
    if arm_id == "ac_zworld_shaped_rl":
        tguard = fan.train_zworld_ac_shaped(
            agent, train_env, seed=seed, n_episodes=rl_eps, steps=steps,
            arm_label=arm_id, denom=rl_eps, shaping_coef=fan.SHAPING_COEF,
        )
        extra["mean_train_forage_recent"] = tguard["mean_train_forage_recent"]
    else:  # ac_zworld_bc
        bguard = fan.bc_warmup_zworld(
            agent, train_env, seed=seed, n_bc=bc_eps, steps=steps,
            arm_label=arm_id, denom=bc_eps,
        )
        extra["bc_action_match_accuracy_recent"] = bguard["bc_action_match_accuracy_recent"]

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(x742.ActorCriticEvalPolicy(agent, arm_id), eval_env, eval_eps, steps)
    row.update(extra)
    row["world_encoder_max_abs_delta"] = float(guard.get("world_encoder_max_abs_delta", 0.0))
    row["zworld_encoder_trained"] = bool(guard.get("zworld_encoder_trained", False))
    row["zworld_p0_episodes"] = int(zworld_p0)
    row["_encoder_guard_report"] = guard  # consumed post-hoc for the worst-cell precondition
    return row


def run_experiment(seeds: List[int], p0: int, zworld_p0: int, rl_eps: int, bc_eps: int,
                   eval_eps: int, steps: int, dry_run: bool) -> Dict[str, Any]:
    print(
        f"MECH-457 H-rep precautionary re-derivation (zworld_p0-trained instrument) "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, zworld_p0={zworld_p0}, RL={rl_eps}, BC={bc_eps}, eval={eval_eps}, "
        f"steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_bc_acc: Dict[str, List[float]] = {"ac_zworld_bc": []}
    per_arm_trainforage: Dict[str, List[float]] = {"ac_zworld_shaped_rl": []}
    treatment_guard_reports: List[Dict[str, Any]] = []
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        slice_cfg = _arm_config_slice(arm_id, env_kwargs, p0, zworld_p0, rl_eps, bc_eps,
                                      eval_eps, steps)
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in TREATMENT_ARMS:
                row = _run_treatment_cell(arm_id, env_kwargs, seed, p0, zworld_p0, rl_eps,
                                          bc_eps, eval_eps, steps, dry_run)
                treatment_guard_reports.append(row.pop("_encoder_guard_report"))
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
    anchors_ready = bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )

    if anchors_ready:
        for arm_id in TREATMENT_ARMS:
            for seed in seeds:
                _run_cell(arm_id, seed)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping treatment training -> substrate_not_ready_requeue", flush=True,
        )

    # Worst-cell encoder-trained guard across BOTH treatment arms x seeds: every treatment cell
    # in this design depends on the SAME zworld_p0=60 warmup training the encoder, so one flat
    # gating precondition (not per-arm scoping) is the correct shape here.
    if treatment_guard_reports:
        worst_guard = min(
            treatment_guard_reports,
            key=lambda r: float(r.get("world_encoder_max_abs_delta", 0.0)),
        )
    else:
        worst_guard = {"world_encoder_max_abs_delta": 0.0, "zworld_encoder_trained": False,
                       "guard_checked": False, "p0_episodes": int(zworld_p0)}
    encoder_precondition = zworld_precondition(
        worst_guard, context="V3-EXQ-748a treatment cells (ac_zworld_shaped_rl + ac_zworld_bc)",
    )
    encoder_ready = bool(encoder_precondition["met"])

    shaped = fan.summarize(per_arm_forage["ac_zworld_shaped_rl"])
    bc = fan.summarize(per_arm_forage["ac_zworld_bc"])
    shaped_maj = bool(shaped["majority_supra_floor"])
    bc_maj = bool(bc["majority_supra_floor"])
    any_dense_maj = bool(shaped_maj or bc_maj)

    if not anchors_ready:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif not encoder_ready:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_dense_maj:
        outcome, label = "PASS", "hrep_elimination_confirmed_under_trained_instrument"
    else:
        outcome, label = "FAIL", "hrep_reopened_under_trained_instrument"

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
             "direction": "lower",
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
            encoder_precondition,
        ],
        "criteria": [
            {"name": "C_any_zworld_dense_arm_clears_floor_at_D3", "load_bearing": True,
             "passed": bool(any_dense_maj)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "zworld_encoder_trained_worst_cell": bool(encoder_ready),
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
            "anchors_ready": anchors_ready,
            "encoder_ready": encoder_ready,
            "local_view_greedy_d3": round(local_view_mean, 6),
            "greedy_oracle_d3": round(oracle_mean, 6),
            "world_encoder_max_abs_delta_worst": round(
                float(worst_guard.get("world_encoder_max_abs_delta", 0.0)), 9
            ),
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
            "reference_748_cotrain_zworld_dense_forage": "748 PASS (cited, not re-run; cotrain "
                "path, structurally unconfounded per failure_autopsy_V3-EXQ-819a)",
        },
        "bootstrap_guard": {
            "load_bearing": False,
            "d3_shaped_mean_train_forage_recent": shaped_train_mean,
            "d3_shaped_eval_foraging": shaped["foraging_competence_mean"],
            "d3_bc_action_match_accuracy_recent": bc_acc_mean,
            "note": (
                "bc_action_match_accuracy is a DIRECT z_world action-adequacy readout under the "
                "zworld_p0-trained, frozen representation: a low fit means this specific "
                "instrument cannot reproduce the competent expert's action from z_world."
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
            "precautionary_rederivation_of": "V3-EXQ-748 (MECH-457 GOV-FANOUT-1 leg B, H-explore)",
            "supersedes": None,
            "reason_not_superseding": (
                "748's PASS is structurally unconfounded by the V3-EXQ-780 frozen-projection "
                "defect (748 used cotrain_encoder=True, not the detached zworld_p0-warmup-then-"
                "frozen mechanism); this run is additional confirmatory hardening, not a "
                "correction."
            ),
            "siblings": ["V3-EXQ-747 (H-rep, raw-view)", "V3-EXQ-749 (conjunction)"],
            "instrument_change": (
                "cotrain_encoder=True (748) -> zworld_p0=60 P0a prediction-training + frozen "
                "(748a), matching V3-EXQ-819a's validated operating point"
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 z_world (zworld_p0=60, then frozen) actor-critic foraging_competence under a "
            "DENSE teacher (shaping OR BC), unshaped eval, vs the 1.0 floor, strict majority of "
            "seeds; readiness = local_view_greedy + oracle clear the floor @D3 AND the P0a "
            "warmup trained the world encoder (worst treatment cell)."
        ),
        "notes": (
            "PRECAUTIONARY RE-DERIVATION, not a live discrimination (DIAGNOSTIC; excluded from "
            "scoring; route to /failure-autopsy before any governance action). GOV-REUSE-1: "
            "queried reanalysis_query.py across all recorded manifests by substrate_hash -- no "
            "manifest carries dense-teacher z_world-BC/shaped-RL under a zworld_p0-trained "
            "encoder (748/747/749/750/751 share the pre-SD-070 substrate_hash "
            "2eff4545309a504f; 819/819a run post-SD-070 but under a SPARSE teacher, not this "
            "dense design) -> NOT RECOVERABLE, run. RE-DERIVE BRAKE: MECH-457 carries 8 counted "
            "substrate_ceiling/non_contributory autopsies (746c-756, fanout-751-750, "
            "fanout-752-753-754, fanout-755, fanout-770-771-772, "
            "gov-fanout-1-cluster-780-781-782, V3-EXQ-765, V3-EXQ-769), none of which is a "
            "same-axis re-pose of the 747/748/749 dense-teacher factorial; brake released on "
            "'substrate now built' grounds -- SD-070's zworld_p0 recipe is IMPLEMENTED (ree-v3 "
            "CLAUDE.md, 2026-07-18/20) and VALIDATED (V3-EXQ-819a, 2026-07-27/30). DV-SYMMETRY: "
            "the manipulation (trained-then-frozen vs co-trained representation) is not a "
            "broadcast constant / monotone rescaling / permutation, so foraging_competence is "
            "not invariant under it by construction (both arms)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-748a MECH-457 H-rep precautionary re-derivation (zworld_p0 instrument)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0, zworld_p0 = fan.DRY_P0, DRY_ZWORLD_P0
        rl_eps, bc_eps = fan.DRY_RL, fan.DRY_BC
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0, zworld_p0 = fan.P0_WARMUP_EPISODES, ZWORLD_P0_EPISODES
        rl_eps, bc_eps = fan.RL_EPISODES, fan.BC_EPISODES
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, zworld_p0=zworld_p0, rl_eps=rl_eps,
                            bc_eps=bc_eps, eval_eps=eval_eps, steps=steps,
                            dry_run=bool(args.dry_run))

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "p0_warmup_episodes": p0, "zworld_p0_episodes": zworld_p0,
        "rl_episodes": rl_eps, "bc_episodes": bc_eps,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN, "cotrain_encoder": False,
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
        f"anchors_ready={result['readiness']['anchors_ready']} "
        f"encoder_ready={result['readiness']['encoder_ready']} "
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
