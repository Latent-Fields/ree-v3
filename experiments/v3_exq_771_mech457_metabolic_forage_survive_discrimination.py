#!/opt/local/bin/python3
"""V3-EXQ-771 -- MECH-457 GOV-FANOUT-1 H2: env reward-coupling (metabolic forage-to-survive).

DIAGNOSTIC discrimination probe (experiment_purpose=diagnostic; claim_ids=["MECH-457"] tags
relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication. MECH-457 stays candidate / v3_pending.

PORTFOLIO CONTEXT (GOV-FANOUT-1, routed by failure_autopsy_V3-EXQ-769_2026-07-17). V3-EXQ-769
EMPIRICALLY FALSIFIED the capacity axis of the competence bootstrap-explorer (256/5x REGRESSED
raw ON foraging 6.48->0.12). Signature: every ON arm survives 200 steps / death 0 / forages ~0
= AVOIDANCE learned WITHOUT APPROACH. KEY MECHANISM (discovered 2026-07-17): on D3_hazard_free
the passive-survival optimum is NOT "do nothing" -- a stay/random agent dies within ~40 steps
from SELF-CONTAMINATION (footprint over contamination_threshold -> contaminated_harm=0.4 drains
health; ~3 contaminated contacts kill). So the trained ON arm survives by learning to keep
MOVING to fresh cells (avoid self-contamination) while IGNORING resources -- survival is coupled
to contamination-AVOIDANCE, not to foraging, and once the intrinsic drive anneals there is no
gradient toward forage. Three competing axes; this is the H2 leg:
  H1 (axis=drive)         -- V3-EXQ-770: the intrinsic drive anneals before approach is set.
  H2 (this, axis=environment) -- V3-EXQ-771: survival is DECOUPLED from foraging.
  H3 (axis=measurement)   -- V3-EXQ-772: the actor-critic cannot assign credit from sparse forage.

THIS LEG (H2, env reward-coupling). Recouple survival from contamination-avoidance to foraging
and ask whether competent foraging then EMERGES. Three arms (both representations, z_world
detached + raw 5x5) at the NON-REGRESSED reference capacity (128-wide, 3x budget, z_world
DETACHED, credit-replay 3/topk 32 -- NOT the 769-falsified 256/5x):
  * CONTROL (ctrl)     -- composed bootstrap on the STANDARD D3 env (contamination ON, no
    starvation). Reproduces the passive-survival plateau: survives via learned contamination-
    avoidance, forages ~0. This is the 765/769 signature, reproduced in-run.
  * DECOUPLE (decouple) -- composed bootstrap on a CONTAMINATION-OFF env (contaminated_harm=0.0)
    with NO starvation coupling. Survival is now TRIVIAL (nothing kills the agent) and STILL
    uncoupled from foraging. This is the ALIASING CONTROL: it isolates whether merely removing
    the contamination distractor (not the reward-coupling) produces foraging.
  * TREATMENT (treat)  -- composed bootstrap on the METABOLIC env: contamination-off PLUS an
    energy->health STARVATION coupling (MetabolicForageWrapper), so an agent that does not
    forage starves and dies while a competent forager (resources respawn on consume) keeps
    energy up and survives. Survival now REQUIRES foraging. Verified 2026-07-17: oracle/
    local_view forage ~60 and survive 200 (death 0); stay/random die (death 1.0).

DECLARED NULL (so a null leg is informative, not wasted):
  * POSITIVE self-route: treat clears the ~13.05 lift-competence target (5.22 plateau + 7.83
    margin) AND beats the DECOUPLE arm by >= the lift margin, on a strict majority of seeds, on
    EITHER representation -> coupling survival to foraging (not merely removing contamination)
    produces competence; env REWARD-COUPLING IS the operative axis. SELF-ROUTE:
    metabolic_coupling_clears_lift_competent.
  * ALIASING catch: if DECOUPLE also clears the lift target (treat - decouple < margin), the
    lift is driven by CONTAMINATION-REMOVAL, not the reward-coupling -> H2's causal claim is
    aliased; the real lever is removing the avoidance distractor. SELF-ROUTE:
    metabolic_lift_aliased_by_contamination_removal (a distinct, informative outcome).
  * NULL: even when survival requires foraging, treat does NOT reach competent foraging (no
    lift over decouple) on BOTH representations -> reward-decoupling is NOT the wall; the wall
    is elsewhere (drive-schedule H1, credit-horizon H3, or deeper). SELF-ROUTE:
    reward_coupling_not_the_axis. All three route to /failure-autopsy to read the portfolio.

READINESS (P0 readiness-assert; SAME statistic as the verdict = foraging_competence vs the 1.0
floor, measured on the METABOLIC env where the treat verdict lives). LocalViewGreedyPolicy +
greedy_oracle must clear the floor on the metabolic env (survival-by-foraging achievable from
the learner's own 5x5 observation). Below readiness -> substrate_not_ready_requeue (FAIL;
NEVER a substrate-verdict label); boot training skipped.

MINT (mint-as-you-go). All ctrl/decouple/treat arms emit reuse-ELIGIBLE per representation
(rng_fully_reset via arm_cell + config_slice_declared + include_driver_script_in_hash=False);
mechanism + probe-env logic lives in experiments/_lib/** (in the substrate hash).

SUBSTRATE STATUS. The metabolic coupling is an experiment-layer PROBE scaffold
(experiments/_lib/mech457_probe_envs.MetabolicForageWrapper), NOT a ree_core feature, per
failure_autopsy_V3-EXQ-769 recommended_substrate_queue_entry.action == "none". If H2 WINS, the
proper native energy->survival coupling routes to /implement-substrate with contract tests.

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
  note: The metabolic coupling makes non-foraging lethal (energy->health starvation), but the
    env is escapable BY FORAGING (readiness confirms oracle/local_view survive), resources
    respawn, and V3 carries no self-model / suffering-like accumulator in this path (SENT-0
    pre-ethical instrumentation). All flags false; decision allow.

SLEEP DRIVER: none (no sleep loop; use_sleep_loop / sws_enabled / rem_enabled all OFF).

Shared machinery: experiments/_lib/mech457_bootstrap_explorer.py + mech457_explorer_classes.py
+ mech457_fanout.py + mech457_probe_envs.py (MetabolicForageWrapper / metabolic_env_kwargs).
ASCII-only in all runtime strings.
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
import experiments._lib.mech457_bootstrap_explorer as boot  # noqa: E402
import experiments._lib.mech457_explorer_classes as mech  # noqa: E402
import experiments._lib.mech457_fanout as fan  # noqa: E402
import experiments._lib.mech457_probe_envs as probe  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_771_mech457_metabolic_forage_survive_discrimination"
QUEUE_ID = "V3-EXQ-771"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

LIFT_COMPETENCE_TARGET = boot.LIFT_COMPETENCE_TARGET   # ~13.05
LIFT_MARGIN = boot.LIFT_ABOVE_PLATEAU                  # ~7.83 (treat must beat decouple by >= this)
RND_PLATEAU = boot.RND_PLATEAU_5_22                    # 5.22

# Reference (non-regressed) composed-bootstrap capacity; held FIXED across all arms. The env
# is the ONLY manipulation across ctrl/decouple/treat.
REF_ACTOR_CRITIC_HIDDEN = fan.ACTOR_CRITIC_HIDDEN      # 128
REF_BUDGET_MULTIPLIER = 3                              # 3x (== 765)
REF_CREDIT_PASSES = mech.CREDIT_REPLAY_PASSES         # 3
REF_CREDIT_TOPK = mech.CREDIT_TOPK                    # 32
REF_COTRAIN_ENCODER = False                           # z_world DETACHED

REPRESENTATIONS: Tuple[str, ...] = ("z_world", "raw_view")
CFG_KINDS: Tuple[str, ...] = ("ctrl", "decouple", "treat")
_REP_TAG = {"z_world": "zworld", "raw_view": "raw"}


def _arm_id(cfg_kind: str, rep: str) -> str:
    return f"metab_{cfg_kind}_{_REP_TAG[rep]}"


BOOT_ARMS: Tuple[str, ...] = tuple(_arm_id(c, r) for r in REPRESENTATIONS for c in CFG_KINDS)
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


def _make_cfg(on_budget: int) -> boot.BootstrapExplorerConfig:
    """The reference composed bootstrap (annealed drive), identical for all three arms -- the
    env is the manipulation, not the config."""
    return boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0, intrinsic_coef_end=boot.ON_INTRINSIC_COEF_END,
        anneal_fraction=boot.ON_ANNEAL_FRACTION,
        warm_start_fraction=0.0,
        entropy_beta_start=boot.ON_ENTROPY_BETA_START, entropy_beta_end=boot.ON_ENTROPY_BETA_END,
        credit_replay=True, credit_replay_passes=REF_CREDIT_PASSES, credit_topk=REF_CREDIT_TOPK,
        n_episodes=int(on_budget),
        actor_critic_hidden=REF_ACTOR_CRITIC_HIDDEN, cotrain_encoder=REF_COTRAIN_ENCODER,
    )


def _arm_env_kwargs(cfg_kind: str, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """ctrl -> standard D3 (contamination ON). decouple/treat -> contamination OFF."""
    if cfg_kind == "ctrl":
        return dict(base_kwargs)
    return probe.metabolic_env_kwargs(base_kwargs)  # contaminated_harm = 0.0


def _make_arm_env(cfg_kind: str, seed: int, base_kwargs: Dict[str, Any]) -> Any:
    """Construct the (possibly wrapped) env for one arm. treat wraps with the starvation
    coupling; ctrl/decouple use the bare env (contamination on / off respectively)."""
    env = x734._make_env(seed, _arm_env_kwargs(cfg_kind, base_kwargs))
    if cfg_kind == "treat":
        return probe.MetabolicForageWrapper(env)
    return env


def _config_slice(arm_id: str, cfg_kind: str, rep: str, cfg: boot.BootstrapExplorerConfig,
                  base_kwargs: Dict[str, Any], p0: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID,
        "env_kwargs": _arm_env_kwargs(cfg_kind, base_kwargs),
        "metabolic_wrapper": bool(cfg_kind == "treat"),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
        "kind": "metabolic_reward_coupling_discrimination", "representation": rep,
        "p0_warmup_episodes": int(p0) if rep == "z_world" else 0,
    }
    base.update(cfg.as_slice())
    if cfg_kind == "treat":
        base.update(probe.config_slice_extra())
    return base


def _run_boot_cell(cfg_kind: str, rep: str, base_kwargs: Dict[str, Any], seed: int, p0: int,
                   on_budget: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind, rep)
    cfg = _make_cfg(on_budget)

    warm_env = _make_arm_env(cfg_kind, seed, base_kwargs)
    rep_agent = mech.make_rep(
        rep, warm_env, seed=seed, p0=p0, steps=steps,
        actor_critic_hidden=int(cfg.actor_critic_hidden),
        cotrain_encoder=bool(cfg.cotrain_encoder),
    )
    train_env = _make_arm_env(cfg_kind, seed, base_kwargs)
    guard = boot.train_bootstrap_explorer(
        rep_agent, train_env, seed=seed, steps=steps, arm_label=arm_id, cfg=cfg,
        denom=cfg.n_episodes,
    )
    eval_env = _make_arm_env(cfg_kind, seed, base_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


def run_experiment(seeds: List[int], p0: int, on_budget: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 GOV-FANOUT-1 H2 metabolic forage-to-survive discrimination "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, ON_budget={on_budget}, eval={eval_eps}, steps={steps}; "
        f"ref_hidden={REF_ACTOR_CRITIC_HIDDEN}, budget_mult={REF_BUDGET_MULTIPLIER}, "
        f"z_world_detached={not REF_COTRAIN_ENCODER}; ctrl=standard decouple=contam_off "
        f"treat=metabolic[energy_decay={probe.METABOLIC_ENERGY_DECAY},"
        f"health_drain={probe.STARVATION_HEALTH_DRAIN}])",
        flush=True,
    )
    base_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    # Readiness anchors + treat's denominators live on the METABOLIC env (where treat forages).
    anchor_kwargs = probe.metabolic_env_kwargs(base_kwargs)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_trainforage: Dict[str, List[float]] = {a: [] for a in BOOT_ARMS}
    per_arm_death: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str], rep: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        if arm_id in BOOT_ARMS:
            cfg = _make_cfg(on_budget)
            slice_cfg = _config_slice(arm_id, cfg_kind, rep, cfg, base_kwargs, p0, eval_eps, steps)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID,
                         "env_kwargs": dict(anchor_kwargs), "metabolic_wrapper": True,
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor_metabolic_env"}
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in BOOT_ARMS:
                row = _run_boot_cell(cfg_kind, rep, base_kwargs, seed, p0, on_budget, eval_eps, steps)
            else:
                # Anchors eval on the metabolic env (the treat-verdict denominator env).
                anchor_env = probe.MetabolicForageWrapper(x734._make_env(seed, anchor_kwargs))
                row = fan.run_anchor_cell(arm_id, anchor_env, seed, eval_eps, steps)
            row["rung_id"] = fan.RUNG_ID
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            cell.stamp(row)
        forage = float(row["foraging_competence"])
        per_arm_forage[arm_id].append(forage)
        per_arm_death[arm_id].append(float(row.get("death_rate", 0.0)))
        if arm_id in BOOT_ARMS:
            per_arm_trainforage[arm_id].append(float(row.get("mean_train_forage_recent", 0.0)))
        all_cells.append(row)
        print(
            f"verdict: {'PASS' if row['competence_supra_floor'] else 'FAIL'} "
            f"(arm={arm_id} seed={seed} forage/ep={forage} death_rate={row.get('death_rate', 0.0)})",
            flush=True,
        )
        return row

    for arm_id in fan.ANCHOR_ARMS:
        for seed in seeds:
            _run_cell(arm_id, seed, None, None)

    def _mean(arm: str) -> float:
        vals = per_arm_forage[arm]
        return float(sum(vals) / len(vals)) if vals else 0.0

    def _mean_death(arm: str) -> float:
        vals = per_arm_death[arm]
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
            f"readiness UNMET on metabolic env (local_view={local_view_mean} oracle={oracle_mean}); "
            f"skipping boot training -> substrate_not_ready_requeue", flush=True,
        )

    # ---- per-representation verdict: treat clears target AND beats DECOUPLE (aliasing control) by margin ----
    def _summ(arm: str) -> Dict[str, Any]:
        return fan.summarize(per_arm_forage[arm])

    per_rep: Dict[str, Any] = {}
    any_rep_clears_lift = False
    any_rep_aliased = False
    for rep in REPRESENTATIONS:
        ctrl_arm, decouple_arm, treat_arm = (
            _arm_id("ctrl", rep), _arm_id("decouple", rep), _arm_id("treat", rep))
        ctrl_s, decouple_s, treat_s = _summ(ctrl_arm), _summ(decouple_arm), _summ(treat_arm)
        ctrl_mean = float(ctrl_s["foraging_competence_mean"])
        decouple_mean = float(decouple_s["foraging_competence_mean"])
        treat_mean = float(treat_s["foraging_competence_mean"])
        treat_per_seed = list(treat_s["foraging_competence_per_seed"])
        n_treat_supra_target = int(sum(1 for v in treat_per_seed if float(v) >= LIFT_COMPETENCE_TARGET))
        strict_majority = bool(n_treat_supra_target > (len(treat_per_seed) / 2.0)) if treat_per_seed else False
        beats_decouple_by_margin = bool((treat_mean - decouple_mean) >= LIFT_MARGIN)
        decouple_also_clears = bool(decouple_mean >= LIFT_COMPETENCE_TARGET)  # aliasing signature
        rep_clears = bool(strict_majority and beats_decouple_by_margin)
        rep_aliased = bool(strict_majority and decouple_also_clears and not beats_decouple_by_margin)
        any_rep_clears_lift = any_rep_clears_lift or rep_clears
        any_rep_aliased = any_rep_aliased or rep_aliased
        per_rep[rep] = {
            "ctrl_arm": ctrl_arm, "decouple_arm": decouple_arm, "treat_arm": treat_arm,
            "ctrl_forage_mean": round(ctrl_mean, 6), "ctrl_forage_per_seed": ctrl_s["foraging_competence_per_seed"],
            "decouple_forage_mean": round(decouple_mean, 6),
            "decouple_forage_per_seed": decouple_s["foraging_competence_per_seed"],
            "treat_forage_mean": round(treat_mean, 6), "treat_forage_per_seed": treat_per_seed,
            "treat_minus_decouple": round(treat_mean - decouple_mean, 6),
            "treat_minus_ctrl": round(treat_mean - ctrl_mean, 6),
            "treat_clears_lift_target_strict_majority": strict_majority,
            "treat_beats_decouple_by_lift_margin": beats_decouple_by_margin,
            "decouple_also_clears_lift_target": decouple_also_clears,
            "rep_clears_lift_competent": rep_clears,
            "rep_lift_aliased_by_contamination_removal": rep_aliased,
            "n_treat_seeds_supra_lift_target": n_treat_supra_target,
            "ctrl_death_rate": round(_mean_death(ctrl_arm), 6),
            "decouple_death_rate": round(_mean_death(decouple_arm), 6),
            "treat_death_rate": round(_mean_death(treat_arm), 6),
        }

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_rep_clears_lift:
        outcome, label = "PASS", "metabolic_coupling_clears_lift_competent"
    elif any_rep_aliased:
        outcome, label = "FAIL", "metabolic_lift_aliased_by_contamination_removal"
    else:
        outcome, label = "FAIL", "reward_coupling_not_the_axis"

    degeneracy = check_degeneracy({
        "boot_arm_and_anchor_foraging": {
            "values": [_mean(a) for a in BOOT_ARMS] + [local_view_mean, _mean("random_walk")]
        }
    })

    interpretation = {
        "label": label,
        "preconditions": [
            {"name": "local_view_greedy_clears_floor_on_metabolic_env", "kind": "readiness",
             "description": "Metabolic env is floor-achievable from the learner's 5x5 local view.",
             "control": "local_view_greedy foraging_competence on metabolic env vs the 1.0 floor",
             "measured": round(local_view_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR)},
            {"name": "greedy_oracle_clears_floor_on_metabolic_env", "kind": "readiness",
             "description": "Metabolic env is floor-achievable with global info (achievability).",
             "control": "greedy_oracle foraging_competence on metabolic env vs the 1.0 floor",
             "measured": round(oracle_mean, 6), "threshold": float(COMPETENCE_RESOURCE_FLOOR),
             "met": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR)},
        ],
        "criteria": [
            {"name": "C_metabolic_coupling_clears_lift_competent_either_rep", "load_bearing": True,
             "passed": bool(any_rep_clears_lift)},
        ],
        "criteria_non_degenerate": {
            "local_view_clears_floor_on_metabolic_env": bool(local_view_mean >= COMPETENCE_RESOURCE_FLOOR),
            "oracle_clears_floor_on_metabolic_env": bool(oracle_mean >= COMPETENCE_RESOURCE_FLOOR),
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
            "local_view_greedy_metabolic": round(local_view_mean, 6),
            "greedy_oracle_metabolic": round(oracle_mean, 6),
        },
        "headline": {
            "any_rep_metabolic_coupling_clears_lift_competent": bool(any_rep_clears_lift),
            "any_rep_lift_aliased_by_contamination_removal": bool(any_rep_aliased),
            "per_representation": per_rep,
            "lift_competence_target": LIFT_COMPETENCE_TARGET,
            "lift_margin": round(LIFT_MARGIN, 6),
            "metabolic_env_local_view_greedy_denominator": round(local_view_mean, 6),
            "metabolic_env_greedy_oracle": round(oracle_mean, 6),
            "metabolic_env_random_walk": round(_mean("random_walk"), 6),
            "metabolic_env_random_walk_death_rate": round(_mean_death("random_walk"), 6),
            "reference_rnd_plateau_751": RND_PLATEAU,
        },
        "per_arm": {a: _summ(a) for a in ARM_ORDER},
        "per_arm_train_forage_recent": {a: _tf(a) for a in BOOT_ARMS},
        "per_arm_death_rate": {a: round(_mean_death(a), 6) for a in ARM_ORDER},
        "reference_band": boot.reference_band(),
        "denominators": {
            "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
            "metabolic_local_view_greedy_live": round(local_view_mean, 6),
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
        "per_arm_death_rate": result["per_arm_death_rate"],
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
                "All ctrl/decouple/treat boot arms emitted reuse-eligible per representation "
                "(rng_fully_reset via arm_cell + config_slice_declared + "
                "include_driver_script_in_hash=False). Mechanism + probe-env logic lives in "
                "experiments/_lib/** (in the substrate hash)."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 foraging_competence of the composed bootstrap on the METABOLIC env (contamination "
            "off + energy->health starvation coupling; survival REQUIRES foraging) vs the "
            "DECOUPLE aliasing control (contamination off, no starvation) and the standard-env "
            "ctrl plateau, at REFERENCE capacity (128-wide/3x/detached), on BOTH z_world and raw. "
            "Verdict: treat clears the ~13.05 lift-competence target AND beats DECOUPLE by >= the "
            "7.83 lift margin, strict-majority of seeds, EITHER rep; readiness = local_view_greedy "
            "+ oracle clear the 1.0 floor on the metabolic env."
        ),
        "notes": (
            "MECH-457 GOV-FANOUT-1 H2 (env reward-coupling) discrimination, routed by "
            "failure_autopsy_V3-EXQ-769_2026-07-17. DIAGNOSTIC (excluded from scoring); "
            "PROMOTES/DEMOTES NOTHING; route to /failure-autopsy (read jointly with H1 V3-EXQ-770 "
            "+ H3 V3-EXQ-772). Manipulation = the ENV only: ctrl=standard D3 (contamination-"
            "avoidance survival, forage ~0 plateau), decouple=contamination-off-no-starvation "
            "(ALIASING control -- isolates contamination-removal from reward-coupling), "
            "treat=metabolic (contamination-off + energy->health starvation; survival REQUIRES "
            "foraging). Capacity/converter/budget/drive held at the NON-REGRESSED reference "
            "(128/3x/detached; NOT the 769-falsified 256/5x). RE-DERIVE BRAKE: PERMITS -- new EXQ "
            "NUMBER on a DIFFERENT design axis (environment), the GOV-FANOUT-1 sanctioned route. "
            "ALIASING catch: decouple clearing the target too (treat-decouple < margin) self-"
            "routes metabolic_lift_aliased_by_contamination_removal. NULL (treat no lift over "
            "decouple both reps) -> reward-coupling NOT the axis. Metabolic coupling is an "
            "experiment-layer PROBE (mech457_probe_envs.MetabolicForageWrapper), NOT a ree_core "
            "feature (769 autopsy substrate action=none); an H2 WIN routes /implement-substrate "
            "for the native coupling. MECH-457 stays candidate/v3_pending."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-771 MECH-457 GOV-FANOUT-1 H2 metabolic forage-to-survive discrimination"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0 = fan.DRY_P0
        on_budget = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0 = fan.P0_WARMUP_EPISODES
        on_budget = int(fan.RL_EPISODES * REF_BUDGET_MULTIPLIER)  # 3000 -- reference budget
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, on_budget=on_budget, eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representations": list(REPRESENTATIONS),
        "p0_warmup_episodes": p0, "on_budget_episodes": on_budget,
        "budget_multiplier": REF_BUDGET_MULTIPLIER,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "ref_actor_critic_hidden": REF_ACTOR_CRITIC_HIDDEN,
        "ref_credit_replay_passes": REF_CREDIT_PASSES, "ref_credit_topk": REF_CREDIT_TOPK,
        "ref_cotrain_encoder": REF_COTRAIN_ENCODER,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "boot_config": _make_cfg(on_budget).as_slice(),
        "probe_env_constants": probe.config_slice_extra(),
        "lift_competence_target": LIFT_COMPETENCE_TARGET, "lift_margin": round(LIFT_MARGIN, 6),
        "competence_resource_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "portfolio": "GOV-FANOUT-1 MECH-457 (H2 env reward-coupling)",
        "routed_by": "failure_autopsy_V3-EXQ-769_2026-07-17",
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
            f"  {rep}: ctrl={pr['ctrl_forage_mean']} decouple={pr['decouple_forage_mean']} "
            f"treat={pr['treat_forage_mean']} (treat-decouple={pr['treat_minus_decouple']}; "
            f"clears_lift={pr['rep_clears_lift_competent']} aliased={pr['rep_lift_aliased_by_contamination_removal']})",
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
