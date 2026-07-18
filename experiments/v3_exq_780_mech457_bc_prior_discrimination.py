#!/opt/local/bin/python3
"""V3-EXQ-780 -- MECH-457 GOV-FANOUT-1 H-bc-prior: competence-directed BEHAVIORAL-PRIOR
(imitation) discrimination.

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PORTFOLIO CONTEXT (GOV-FANOUT-1, routed by failure_autopsy_MECH-457-fanout-770-771-772_2026-
07-18). The 770/771/772 discrimination portfolio (drive-schedule / reward-coupling / credit-
horizon) all FAILED, each hitting its pre-registered null; combined with V3-EXQ-769 (capacity
falsified) FOUR axes are now eliminated. The learned actor-critic converter cannot extract a
competent foraging policy from a provably-sufficient observation (local_view_greedy on the
learner's own 5x5 view forages 48-55; oracle 57-61; every treatment arm forages at the ~1.0
floor), invariant to drive/reward/credit/capacity. Missing-dependency reframe (user-confirmed):
the actor needs a COMPETENCE-DIRECTED BOOTSTRAP; BC (imitation, 32.72 in V3-EXQ-748) is the only
floor-clearing existence proof. The re-derive brake FIRED (7th non_contributory MECH-457
autopsy) and REFUSES any further config/env/credit/capacity same-axis re-pose. It PERMITS this
fan-out: NEW EXQ numbers on DIFFERENT mechanism classes (same sanction as 769->770/771/772). Two
legs discriminate WHICH competence-directed dependency:
  H-bc-prior (this, axis=learning-signal)  -- V3-EXQ-780: a LEARNED behavioral prior (imitation).
  H-approach-primitive (axis=intrinsic-arch) -- V3-EXQ-781: an innate non-extinguishing approach
                                              drive, demonstrator-free (avoids aliasing this leg).

THIS LEG (H-bc-prior, learning-signal). Seed the composed bootstrap's actor with a competence-
directed BEHAVIORAL PRIOR distilled from the floor-clearing demonstrator (LocalViewGreedyPolicy,
which forages 48.05 on the SAME 5x5 view the learner reads): a BC-of-policy WARM-START (clone the
demonstrator's actions into the actor's policy BEFORE RL) PLUS a persistent imitation AUXILIARY
(a CE(actor_logits, demo_action) term on the on-policy state distribution during RL, keeping the
seed alive against RL drift). Everything else is held FIXED at the NON-regressed reference build
(128-wide actor, 3x budget, z_world DETACHED, credit-replay 3/topk 32, developmental drive anneal
1.0->0.05) -- deliberately NOT the 769-regressed 256/5x build:
  * CONTROL (ctrl)  -- the reference composed bootstrap, NO BC (== the 770 ctrl; the 765-class
    plateau at reference capacity). bc_aux_coef=0, no warm-start.
  * TREATMENT (treat) -- ctrl + BC warm-start (300 episodes of imitation) + persistent BC
    auxiliary (bc_aux_coef>0). Everything else identical to ctrl.
Both representations (z_world DETACHED + raw 5x5), per the lineage's reuse directive.

COVARIATES (make the null informative + separable, per the autopsy's aliasing audit):
  * post_bc_foraging_competence -- the seed's foraging competence AFTER warm-start, BEFORE RL.
    Separates "the seed never took" (post_bc ~0) from "the seed was ERASED by RL" (post_bc clears
    the floor, post_rl collapses). bc_seed_took / bc_seed_survives_rl booleans record this.
  * raw-vs-z_world -- both reps run so a representation-riding failure (seed survives on raw but
    not z_world) is separable from a bootstrap-content failure.
  * bc_warmstart_action_match_recent / bc_aux_action_match_recent -- imitation fidelity during
    the warm-start and during RL.

DECLARED NULL (so a null leg is informative, not wasted):
  * POSITIVE self-route: treat clears the ~13.05 lift-competence target (5.22 RND plateau + 7.83
    lift margin) AND beats its own ctrl arm by >= the lift margin, on a strict majority of seeds,
    on EITHER representation -> a competence-directed BEHAVIORAL PRIOR lets the actor-critic
    convert a sufficient observation into competent foraging; the missing dependency IS a learned
    imitation prior. SELF-ROUTE: bc_prior_seed_clears_lift_competent.
  * NULL: even imitation-seeded (warm-start + persistent auxiliary), the actor collapses back to
    passive-survival at unshaped eval on BOTH representations -> the deficit is NOT a missing
    behavioral prior but something that ERASES the seeded policy (a stability / representation-
    riding failure), OR the missing piece is the innate approach primitive (H-approach-primitive,
    781). This does NOT weaken MECH-457 (diagnostic; promotes/demotes nothing) but it removes
    H-bc-prior from the live set. SELF-ROUTE: bc_prior_not_the_axis. Routes to /failure-autopsy to
    read the portfolio jointly with 781.

AXIS ISOLATION vs H-approach-primitive (781; anti-aliasing). This leg uses a DEMONSTRATOR (the
learning signal IS supervised imitation of an expert). 781 is DEMONSTRATOR-FREE (an intrinsic
resource-proximity drive, no expert action) -- so a 780-positive/781-null (or vice versa)
cleanly attributes the missing dependency to the learned-imitation axis vs the innate-approach
axis. The two legs sit on DIFFERENT axes (supervised learning-signal vs intrinsic architecture).

READINESS (P0 readiness-assert; SAME statistic as the verdict = foraging_competence @D3 vs the
1.0 floor). LocalViewGreedyPolicy (5x5 view) + greedy_oracle must clear the floor @D3 (env
solvable from the learner's own observation). Below readiness -> substrate_not_ready_requeue
(FAIL; NEVER a substrate-verdict label); boot training skipped.

MINT (mint-as-you-go). Both ctrl and treat arms emit reuse-ELIGIBLE per representation
(rng_fully_reset via arm_cell + config_slice_declared + include_driver_script_in_hash=False);
mechanism logic (BC warm-start + persistent auxiliary) lives in experiments/_lib/** (in the
substrate hash).

evidence_direction = "unknown" (DIAGNOSTIC; verdict lives in interpretation.label /
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

Shared machinery: experiments/_lib/mech457_bootstrap_explorer.py (composed primitive) +
mech457_explorer_classes.py (RepAgent / RNDModule / train_a2c BC-auxiliary hook /
warmstart_bc_rep) + mech457_fanout.py (anchors / readiness / budgets) +
capability_eval.LocalViewGreedyPolicy (the demonstrator). ASCII-only in all runtime strings.
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
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    evaluate_seed,
)
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_780_mech457_bc_prior_discrimination"
QUEUE_ID = "V3-EXQ-780"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# Load-bearing thresholds (pre-registered in the _lib; declared here, not derived from the run).
LIFT_COMPETENCE_TARGET = boot.LIFT_COMPETENCE_TARGET   # ~13.05 (5.22 plateau + 7.83 lift margin)
LIFT_MARGIN = boot.LIFT_ABOVE_PLATEAU                  # ~7.83 (treat must beat ctrl by >= this)
RND_PLATEAU = boot.RND_PLATEAU_5_22                    # 5.22 (reference band centre)

# Reference (non-regressed) composed-bootstrap capacity: the 765-class build MINUS the 769
# capacity regression (256/5x) and MINUS warm-start/credit-boost. Held FIXED across ctrl/treat
# (identical to the 770 reference ctrl). The BC seed is the ONLY manipulation for treat.
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN      # 128 (reference width)
REF_BUDGET_MULTIPLIER = 3                              # 3x the 1000-ep plateau budget (== 765/770)
REF_CREDIT_PASSES = mech.CREDIT_REPLAY_PASSES         # 3 (reference converter)
REF_CREDIT_TOPK = mech.CREDIT_TOPK                    # 32
REF_COTRAIN_ENCODER = False                           # z_world DETACHED (cleaner integration)

# H-bc-prior treatment knobs (design choices, not tuned magic numbers).
BC_AUX_COEF = 0.5                                     # persistent imitation CE weight during RL

REPRESENTATIONS: Tuple[str, ...] = ("z_world", "raw_view")
CFG_KINDS: Tuple[str, ...] = ("ctrl", "treat")
_REP_TAG = {"z_world": "zworld", "raw_view": "raw"}


def _arm_id(cfg_kind: str, rep: str) -> str:
    return f"bcprior_{cfg_kind}_{_REP_TAG[rep]}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(c, r) for r in REPRESENTATIONS for c in CFG_KINDS)
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


def _make_cfg(cfg_kind: str, on_budget: int) -> boot.BootstrapExplorerConfig:
    """Reference composed bootstrap (annealed drive 1.0->0.05, warm-start 0, credit-replay 3/32,
    128-wide, z_world detached) -- identical to the 770 ctrl. treat adds bc_aux_coef>0 (the
    persistent imitation auxiliary); the BC warm-start is applied driver-side. ctrl = bc_aux_coef
    0 (no BC at all). All other knobs identical."""
    if cfg_kind not in CFG_KINDS:
        raise ValueError(f"unknown cfg_kind {cfg_kind!r}")
    bc_aux = BC_AUX_COEF if cfg_kind == "treat" else 0.0
    return boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0, intrinsic_coef_end=boot.ON_INTRINSIC_COEF_END,
        anneal_fraction=boot.ON_ANNEAL_FRACTION,
        warm_start_fraction=0.0,
        entropy_beta_start=boot.ON_ENTROPY_BETA_START, entropy_beta_end=boot.ON_ENTROPY_BETA_END,
        credit_replay=True, credit_replay_passes=REF_CREDIT_PASSES, credit_topk=REF_CREDIT_TOPK,
        n_episodes=int(on_budget),
        actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN, cotrain_encoder=REF_COTRAIN_ENCODER,
        bc_aux_coef=bc_aux,
    )


def _config_slice(arm_id: str, rep: str, cfg: boot.BootstrapExplorerConfig,
                  env_kwargs: Dict[str, Any], p0: int, eval_eps: int, steps: int,
                  bc_warmstart_episodes: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
        "kind": "bc_prior_discrimination", "representation": rep,
        "bc_warmstart_episodes": int(bc_warmstart_episodes),
        "bc_demonstrator": "local_view_greedy",
        "p0_warmup_episodes": int(p0) if rep == "z_world" else 0,
    }
    base.update(cfg.as_slice())
    return base


def _run_boot_cell(cfg_kind: str, rep: str, env_kwargs: Dict[str, Any], seed: int, p0: int,
                   on_budget: int, eval_eps: int, steps: int,
                   bc_warmstart_episodes: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind, rep)
    is_treat = (cfg_kind == "treat")
    cfg = _make_cfg(cfg_kind, on_budget)

    warm_env = x734._make_env(seed, env_kwargs)
    rep_agent = mech.make_rep(
        rep, warm_env, seed=seed, p0=p0, steps=steps,
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
    )

    demo = LocalViewGreedyPolicy(seed=seed) if is_treat else None
    post_bc_forage = 0.0
    bc_warm_match = 0.0
    if is_treat:
        bc_env = x734._make_env(seed, env_kwargs)
        wguard = mech.warmstart_bc_rep(
            rep_agent, bc_env, seed=seed, n_bc=int(bc_warmstart_episodes), steps=steps,
            demo=demo, arm_label=arm_id, denom=int(bc_warmstart_episodes),
        )
        bc_warm_match = float(wguard.get("bc_warmstart_action_match_accuracy_recent", 0.0))
        # post-BC / pre-RL eval: did the seed take? (separates never-took from RL-erased).
        postbc_env = x734._make_env(seed, env_kwargs)
        postbc_row = evaluate_seed(rep_agent.eval_policy(arm_id + "_postbc"), postbc_env, eval_eps, steps)
        post_bc_forage = float(postbc_row["foraging_competence"])

    train_env = x734._make_env(seed, env_kwargs)
    guard = boot.train_bootstrap_explorer(
        rep_agent, train_env, seed=seed, steps=steps, arm_label=arm_id, cfg=cfg,
        denom=cfg.n_episodes, bc_demo=demo,
    )
    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    if is_treat:
        forage = float(row["foraging_competence"])
        row["bc_warmstart_action_match_recent"] = round(bc_warm_match, 6)
        row["bc_aux_action_match_recent"] = float(guard.get("mean_bc_aux_action_match_recent", 0.0))
        row["post_bc_foraging_competence"] = round(post_bc_forage, 6)
        row["bc_seed_took"] = bool(post_bc_forage >= COMPETENCE_RESOURCE_FLOOR)
        row["bc_seed_survives_rl"] = bool(forage >= COMPETENCE_RESOURCE_FLOOR)
    return row


def run_experiment(seeds: List[int], p0: int, on_budget: int, eval_eps: int, steps: int,
                   bc_warmstart_episodes: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H-bc-prior discrimination "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, ON_budget={on_budget}, bc_warmstart={bc_warmstart_episodes}, "
        f"eval={eval_eps}, steps={steps}; ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, "
        f"budget_mult={REF_BUDGET_MULTIPLIER}, z_world_detached={not REF_COTRAIN_ENCODER}; "
        f"treat=BC_warmstart+aux(coef={BC_AUX_COEF}) demonstrator=local_view_greedy)",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_trainforage: Dict[str, List[float]] = {a: [] for a in BOOT_ARMS}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str], rep: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        if arm_id in BOOT_ARMS:
            cfg = _make_cfg(cfg_kind, on_budget)
            slice_cfg = _config_slice(arm_id, rep, cfg, env_kwargs, p0, eval_eps, steps,
                                      bc_warmstart_episodes if cfg_kind == "treat" else 0)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in BOOT_ARMS:
                row = _run_boot_cell(cfg_kind, rep, env_kwargs, seed, p0, on_budget, eval_eps,
                                     steps, bc_warmstart_episodes)
            else:
                anchor_env = x734._make_env(seed, env_kwargs)
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        if arm_id in BOOT_ARMS:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage})", flush=True,
        )
        return row

    # Anchors first (readiness gate + denominators), then the boot arms if ready.
    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed, None, None)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    local_view_mean = _mean("local_view_greedy")
    oracle_mean = _mean("greedy_oracle")
    readiness_met = bool(
        local_view_mean >= COMPETENCE_RESOURCE_FLOOR and oracle_mean >= COMPETENCE_RESOURCE_FLOOR
    )

    if readiness_met:
        for rep in REPRESENTATIONS:
            for cfg_kind in CFG_KINDS:
                arm_id = _arm_id(cfg_kind, rep)
                for seed in seeds:
                    _run_cell(arm_id, seed, cfg_kind, rep)
    else:
        print(
            f"readiness UNMET (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping boot training -> substrate_not_ready_requeue", flush=True,
        )

    # ---- per-representation lift verdict: treat clears the target AND beats ctrl by margin ----
    def _summ(arm: str) -> Dict[str, Any]:
        return fan.summarize(per_arm_forage[arm])

    def _cell_for(arm_id: str) -> List[Dict[str, Any]]:
        return [c for c in all_cells if c.get("arm_id") == arm_id]

    per_rep: Dict[str, Any] = {}
    any_rep_clears_lift = False
    for rep in REPRESENTATIONS:
        ctrl_arm, treat_arm = _arm_id("ctrl", rep), _arm_id("treat", rep)
        ctrl_s, treat_s = _summ(ctrl_arm), _summ(treat_arm)
        ctrl_mean = float(ctrl_s["foraging_competence_mean"])
        treat_mean = float(treat_s["foraging_competence_mean"])
        treat_per_seed = list(treat_s["foraging_competence_per_seed"])
        n_treat_supra_target = int(sum(1 for v in treat_per_seed if float(v) >= LIFT_COMPETENCE_TARGET))
        strict_majority = bool(n_treat_supra_target > (len(treat_per_seed) / 2.0)) if treat_per_seed else False
        beats_ctrl_by_margin = bool((treat_mean - ctrl_mean) >= LIFT_MARGIN)
        rep_clears = bool(strict_majority and beats_ctrl_by_margin)
        any_rep_clears_lift = any_rep_clears_lift or rep_clears
        treat_cells = _cell_for(treat_arm)
        post_bc = [float(c.get("post_bc_foraging_competence", 0.0)) for c in treat_cells]
        n_seed_took = int(sum(1 for c in treat_cells if bool(c.get("bc_seed_took", False))))
        n_seed_survives = int(sum(1 for c in treat_cells if bool(c.get("bc_seed_survives_rl", False))))
        per_rep[rep] = {
            "ctrl_arm": ctrl_arm, "treat_arm": treat_arm,
            "ctrl_forage_mean": round(ctrl_mean, 6), "ctrl_forage_per_seed": ctrl_s["foraging_competence_per_seed"],
            "treat_forage_mean": round(treat_mean, 6), "treat_forage_per_seed": treat_per_seed,
            "treat_minus_ctrl": round(treat_mean - ctrl_mean, 6),
            "treat_clears_lift_target_strict_majority": strict_majority,
            "treat_beats_ctrl_by_lift_margin": beats_ctrl_by_margin,
            "rep_clears_lift_competent": rep_clears,
            "n_treat_seeds_supra_lift_target": n_treat_supra_target,
            # BC covariates (null-separability): seed-took (post-BC pre-RL) vs seed-survives-RL.
            "treat_post_bc_forage_per_seed": [round(v, 6) for v in post_bc],
            "treat_post_bc_forage_mean": round(float(sum(post_bc) / len(post_bc)), 6) if post_bc else 0.0,
            "n_treat_seeds_bc_seed_took": n_seed_took,
            "n_treat_seeds_bc_seed_survives_rl": n_seed_survives,
        }

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_rep_clears_lift:
        outcome, label = "PASS", "bc_prior_seed_clears_lift_competent"
    else:
        outcome, label = "FAIL", "bc_prior_not_the_axis"

    degeneracy = check_degeneracy({
        "d3_boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        }
    })

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
            {"name": "C_bc_prior_clears_lift_competent_either_rep", "load_bearing": True,
             "passed": bool(any_rep_clears_lift)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_at_d3": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_at_d3": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
            "boot_arm_vs_anchor_foraging_spread": bool(degeneracy["non_degenerate"]),
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
            "any_rep_bc_prior_clears_lift_competent": bool(any_rep_clears_lift),
            "per_representation": per_rep,
            "lift_competence_target": LIFT_COMPETENCE_TARGET,
            "lift_margin": round(LIFT_MARGIN, 6),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "reference_rnd_plateau_751": RND_PLATEAU,
            "bc_demonstrator": "local_view_greedy",
        },
        "per_arm": {a: _summ(a) for a in ARM_ORDER},
        "per_arm_train_forage_recent": {a: _tf(a) for a in BOOT_ARMS},
        "reference_band": boot.reference_band(),
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
        "denominators": result["denominators"],
        "per_arm": result["per_arm"],
        "per_arm_train_forage_recent": result["per_arm_train_forage_recent"],
        "reference_band": result["reference_band"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "reuse_mint": {
            "reusable_arms": list(BOOT_ARMS),
            "reuse_eligible": True,
            "note": (
                "All ctrl and treat boot arms emitted reuse-eligible per representation "
                "(rng_fully_reset via arm_cell + config_slice_declared + "
                "include_driver_script_in_hash=False). Mechanism logic (BC warm-start + "
                "persistent imitation auxiliary) lives in experiments/_lib/mech457_explorer_"
                "classes + mech457_bootstrap_explorer (in the substrate hash)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 foraging_competence (unshaped eval) of the BC-SEEDED composed bootstrap (BC "
            "warm-start from local_view_greedy + persistent imitation auxiliary, bc_aux_coef="
            f"{BC_AUX_COEF}) vs the NO-BC reference ctrl, at REFERENCE capacity (128-wide, 3x "
            "budget, z_world detached, credit-replay 3/topk 32), on BOTH z_world and raw 5x5. "
            "Verdict: treat clears the ~13.05 lift-competence target (5.22 plateau + 7.83 "
            "margin) AND beats its ctrl arm by >= the lift margin, on a strict majority of "
            "seeds, on EITHER representation; readiness = local_view_greedy + oracle clear the "
            "1.0 floor @D3. Covariate: post_bc_foraging_competence separates 'seed never took' "
            "from 'seed erased by RL'."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 H-bc-prior (learning-signal / imitation) discrimination, "
            "routed by failure_autopsy_MECH-457-fanout-770-771-772_2026-07-18. DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy (read "
            "jointly with H-approach-primitive V3-EXQ-781). Manipulation = a competence-directed "
            "BEHAVIORAL PRIOR (BC warm-start + persistent imitation auxiliary from local_view_"
            "greedy) added to the NON-REGRESSED reference build (128-wide/3x/detached; NOT the "
            "769-falsified 256/5x). RE-DERIVE BRAKE: PERMITS -- new EXQ NUMBER on a DIFFERENT "
            "mechanism class (learning-signal), the GOV-FANOUT-1 sanctioned route the "
            "770/771/772 autopsy names (same sanction as 769->770/771/772); the brake refuses "
            "only another config/env/credit/capacity same-axis re-pose. Pre-registered ALIVE as "
            "H-bc-prior in hypothesis_space_registry.v1.json (competence_floor). NULL (treat "
            "still forages ~0 both reps) -> a learned behavioral prior is NOT the axis (removes "
            "H-bc-prior), does NOT weaken MECH-457. Anti-alias vs H-approach-primitive (781): "
            "this leg is DEMONSTRATOR-based (supervised imitation); 781 is demonstrator-free "
            "(intrinsic approach drive). MECH-457 stays candidate/v3_pending."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-780 MECH-457 GOV-FANOUT-1 H-bc-prior discrimination"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0 = fan.DRY_P0
        on_budget = fan.DRY_RL
        bc_warmstart = fan.DRY_BC
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0 = fan.P0_WARMUP_EPISODES
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)  # 3000 -- reference budget
        bc_warmstart = fan.BC_EPISODES                            # 300 -- warm-start budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, on_budget=on_budget, eval_eps=eval_eps,
                            steps=steps, bc_warmstart_episodes=bc_warmstart)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representations": list(REPRESENTATIONS),
        "p0_warmup_episodes": p0, "on_budget_episodes": on_budget,
        "bc_warmstart_episodes": bc_warmstart, "bc_aux_coef": BC_AUX_COEF,
        "bc_demonstrator": "local_view_greedy",
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": REF_CREDIT_PASSES, "ref_credit_topk": REF_CREDIT_TOPK,
        "ref_cotrain_encoder": REF_COTRAIN_ENCODER,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA, "bc_lr": fan.BC_LR,
        "ctrl_config": _make_cfg("ctrl", on_budget).as_slice(),
        "treat_config": _make_cfg("treat", on_budget).as_slice(),
        "lift_competence_target": LIFT_COMPETENCE_TARGET, "lift_margin": round(LIFT_MARGIN, 6),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 (H-bc-prior learning-signal)",
        "routed_by": "failure_autopsy_MECH-457-fanout-770-771-772_2026-07-18",
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
    for rep in REPRESENTATIONS:
        pr = hl["per_representation"][rep]
        print(
            f"  {rep}: ctrl={pr['ctrl_forage_mean']} treat={pr['treat_forage_mean']} "
            f"(treat-ctrl={pr['treat_minus_ctrl']}; clears_lift={pr['rep_clears_lift_competent']}; "
            f"post_bc={pr['treat_post_bc_forage_mean']} "
            f"seed_took={pr['n_treat_seeds_bc_seed_took']} "
            f"seed_survives={pr['n_treat_seeds_bc_seed_survives_rl']})",
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
