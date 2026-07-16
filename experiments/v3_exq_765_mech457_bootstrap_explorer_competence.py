#!/opt/local/bin/python3
"""V3-EXQ-765 -- MECH-457 competence bootstrap-explorer validation (post-build retest).

DIAGNOSTIC substrate-readiness validation (experiment_purpose=diagnostic; claim_ids=["MECH-457"]
tags relevance only -> excluded from governance confidence/conflict scoring). PROMOTES / DEMOTES
NOTHING. Routes to /failure-autopsy for adjudication before any governance action. MECH-457 stays
candidate / v3_pending; INV-088 stays candidate / pending_substrate_reconfirmation.

WHY. The MECH-457 GOV-FANOUT-1 discrimination is CLOSED across five autopsies (751-756): no SINGLE
mechanism and no PAIRWISE combination clears the 1.0 foraging floor toward the 48.05 local-view
ceiling. The wall is ONE structural property with two joined halves (failure_autopsy_MECH-457-
fanout-755_2026-07-15 cluster read): (1) cold-start / success-dependence -- credit/return/curriculum
/pair (752/753/754/756) amplify signal derived from prior success -> collapse sub-floor from ~0;
only success-INDEPENDENT RND (751, 5.22) and BC (748, 32.72, needs an expert) break the floor; (2)
capacity-to-convert -- even a competent explore/exploit gate (755) adds nothing over fixed RND; RND
reaches only ~11% of the ceiling. Every fanout CONVERTER (752 credit, 756 pair) ran on the SPARSE
base -> nothing to convert. The genuinely-untested, highest-composition build is the RND drive
COMPOSED WITH a converter + budget + a developmental explore->exploit anneal -- built as the
substrate mech457_competence_bootstrap_explorer (experiments/_lib/mech457_bootstrap_explorer.py,
ree-v3 main 022a284, 2026-07-16). This EXQ is its Step-8 validation.

RE-DERIVE BRAKE: RELEASED. MECH-457 has >=2 non_contributory/convergent autopsies (751-750,
752-753-754, 755, 746c-756), so the brake fired on the single-axis fanout legs. It is RELEASED here
because (a) the named upstream substrate mech457_competence_bootstrap_explorer is now IMPLEMENTED
(ree-v3/CLAUDE.md, 2026-07-16), making the re-test meaningful, and (b) this is a genuinely-DISTINCT
COMPOSED mechanism (new EXQ NUMBER 765, not a lettered single-axis bolt-on). The brake refuses
another single-axis probe, NOT the post-build composed retest.

GOV-REUSE-1: the decisive readout (composed-bootstrap ON foraging_competence vs the 5.22 RND
plateau, both representations) is recorded in NO prior manifest -- 751 ran RND-only (no anneal / no
credit / plateau budget), 752-756 ran converters on the SPARSE base. Not recoverable -> run.

THE ARMS (both representations, per the reuse directive; OFF and ON emitted reuse-ELIGIBLE):
  * boot_off_zworld / boot_off_raw -- make_off_config: RND plateau reproduction (constant coef 1.0,
    no anneal, no credit-replay, plateau budget 1000). The drift-guard: OFF should reproduce the
    ~5.22 RND plateau band. If OFF unexpectedly clears the lift-competence target, the comparison is
    confounded by substrate drift (flag; do NOT self-route).
  * boot_on_zworld / boot_on_raw -- make_on_config: the composed bootstrap (RND drive + prioritized
    credit-replay converter + developmental coef 1.0->0.05 / entropy 0.10->0.03 anneal + 3x budget).
  * anchors (local_view_greedy, greedy_oracle, random_walk) -- readiness gate + denominators, eval
    live (representation-agnostic, read the env directly).

DECLARED NULL (so a sub-target leg is informative, not wasted):
  * ON clears the lift-competence target (~13.05 res/ep = the 5.22 plateau + a 7.83 lift margin) AND
    beats its own OFF arm by >= the lift margin, on a strict majority of seeds, on EITHER
    representation -> the composed bootstrap converts coverage into competence; the substrate is
    validated (floor->competent). SELF-ROUTE: bootstrap_explorer_clears_lift_competent.
  * ON stays at / near the OFF plateau (no lift-margin gain) on BOTH representations -> composing the
    landed pieces at this capacity/budget does NOT bootstrap floor->competent; the residual is a
    DEEPER capacity gap (a bigger policy / longer budget / a different converter), not this
    composition. SELF-ROUTE: bootstrap_explorer_plateaus_capacity_gap_remains. Routes to
    /failure-autopsy to decide the next build (NOT another single-axis probe -- brake still holds
    for those).

READINESS (P0 readiness-assert; SAME statistic as the verdict = foraging_competence @D3 vs the 1.0
floor). LocalViewGreedyPolicy (5x5 view) and greedy_oracle clear the floor @D3 (env solvable). Below
readiness -> substrate_not_ready_requeue (FAIL; NEVER a substrate-verdict label); boot training
skipped.

MINT (mint-as-you-go). Both OFF and ON arms are emitted reuse-ELIGIBLE per representation
(rng_fully_reset via arm_cell + config_slice_declared + include_driver_script_in_hash=False). The
mechanism logic lives in experiments/_lib/** (mech457_bootstrap_explorer + mech457_explorer_classes),
so it is IN the substrate hash and the fingerprint refuses on substrate drift. A later cross-driver
MECH-457/INV-088 consumer -- esp. the V3-EXQ-750 strategy-diversity retest needing matched-competent
policies on BOTH representations -- can cite reuse_baseline_from these cells.

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

Shared machinery: experiments/_lib/mech457_bootstrap_explorer.py (the composed primitive) +
experiments/_lib/mech457_explorer_classes.py (RepAgent / RNDModule / train_a2c) +
experiments/_lib/mech457_fanout.py (anchors / readiness / budgets). ASCII-only in all runtime strings.
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
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_765_mech457_bootstrap_explorer_competence"
QUEUE_ID = "V3-EXQ-765"
CLAIM_IDS: List[str] = ["MECH-457"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

DEVICE = fan.DEVICE

# Load-bearing thresholds (pre-registered in the _lib; declared here, not derived from the run).
LIFT_COMPETENCE_TARGET = boot.LIFT_COMPETENCE_TARGET   # ~13.05 (5.22 plateau + 7.83 lift margin)
LIFT_MARGIN = boot.LIFT_ABOVE_PLATEAU                  # ~7.83 (ON must beat its OFF by >= this)
RND_PLATEAU = boot.RND_PLATEAU_5_22                    # 5.22 (the OFF drift-guard band centre)

REPRESENTATIONS: Tuple[str, ...] = ("z_world", "raw_view")
CFG_KINDS: Tuple[str, ...] = ("off", "on")

# arm_id = boot_<cfg>_<rep_tag>; rep_tag: z_world->zworld, raw_view->raw.
_REP_TAG = {"z_world": "zworld", "raw_view": "raw"}


def _arm_id(cfg_kind: str, rep: str) -> str:
    return f"boot_{cfg_kind}_{_REP_TAG[rep]}"


BOOT_ARMS: Tuple[str, ...] = tuple(
    _arm_id(c, r) for r in REPRESENTATIONS for c in CFG_KINDS
)
ARM_ORDER: Tuple[str, ...] = BOOT_ARMS + fan.ANCHOR_ARMS


def _make_cfg(cfg_kind: str, on_budget: int, off_budget: int) -> boot.BootstrapExplorerConfig:
    if cfg_kind == "off":
        return boot.make_off_config(n_episodes=off_budget)
    cfg = boot.make_on_config()
    cfg.n_episodes = int(on_budget)
    return cfg


def _config_slice(arm_id: str, rep: str, cfg: boot.BootstrapExplorerConfig,
                  env_kwargs: Dict[str, Any], p0: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
        "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
        "kind": "bootstrap_explorer", "representation": rep,
        "cotrain_encoder": bool(rep == "z_world"),
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
        "p0_warmup_episodes": int(p0) if rep == "z_world" else 0,
    }
    base.update(cfg.as_slice())
    return base


def _run_boot_cell(cfg_kind: str, rep: str, env_kwargs: Dict[str, Any], seed: int, p0: int,
                   on_budget: int, off_budget: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    arm_id = _arm_id(cfg_kind, rep)
    cfg = _make_cfg(cfg_kind, on_budget, off_budget)

    warm_env = x734._make_env(seed, env_kwargs)
    rep_agent = mech.make_rep(rep, warm_env, seed=seed, p0=p0, steps=steps)  # z_world warms up here

    train_env = x734._make_env(seed, env_kwargs)
    guard = boot.train_bootstrap_explorer(
        rep_agent, train_env, seed=seed, steps=steps, arm_label=arm_id, cfg=cfg,
        denom=cfg.n_episodes,
    )

    eval_env = x734._make_env(seed, env_kwargs)
    row = evaluate_seed(rep_agent.eval_policy(arm_id), eval_env, eval_eps, steps)
    row["mean_train_forage_recent"] = float(guard.get("mean_train_forage_recent", 0.0))
    row["mean_intrinsic_reward_recent"] = float(guard.get("mean_intrinsic_reward_recent", 0.0))
    row["n_credit_replay_passes"] = int(guard.get("n_credit_replay_passes", 0))
    return row


def run_experiment(seeds: List[int], p0: int, on_budget: int, off_budget: int, eval_eps: int,
                   steps: int) -> Dict[str, Any]:
    print(
        f"MECH-457 competence bootstrap-explorer validation (post-build retest) "
        f"({len(ARM_ORDER)} arms x 1 rung [{fan.RUNG_ID}] x {len(seeds)} seeds; "
        f"P0={p0}, ON_budget={on_budget}, OFF_budget={off_budget}, eval={eval_eps}, steps={steps})",
        flush=True,
    )
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    per_arm_forage: Dict[str, List[float]] = {a: [] for a in ARM_ORDER}
    per_arm_trainforage: Dict[str, List[float]] = {a: [] for a in BOOT_ARMS}
    all_cells: List[Dict[str, Any]] = []

    def _run_cell(arm_id: str, seed: int, cfg_kind: Optional[str], rep: Optional[str]) -> Dict[str, Any]:
        print(f"Seed {seed} Condition {fan.RUNG_ID}:{arm_id}", flush=True)
        if arm_id in BOOT_ARMS:
            cfg = _make_cfg(cfg_kind, on_budget, off_budget)
            slice_cfg = _config_slice(arm_id, rep, cfg, env_kwargs, p0, eval_eps, steps)
        else:
            slice_cfg = {"arm_id": arm_id, "rung_id": fan.RUNG_ID, "env_kwargs": dict(env_kwargs),
                         "eval_episodes": int(eval_eps), "steps_per_episode": int(steps),
                         "kind": "anchor"}
        # Both OFF and ON boot arms mint reuse-eligible (mechanism logic is in _lib = in the
        # substrate hash); anchors are eval-only anchors (also eligible). No shared cross-cell state.
        with arm_cell(seed, config_slice=slice_cfg, script_path=Path(__file__),
                      config_slice_declared=True, include_driver_script_in_hash=False) as cell:
            if arm_id in BOOT_ARMS:
                row = _run_boot_cell(cfg_kind, rep, env_kwargs, seed, p0, on_budget, off_budget,
                                     eval_eps, steps)
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

    # ---- per-representation lift verdict --------------------------------------------------
    def _summ(arm: str) -> Dict[str, Any]:
        return fan.summarize(per_arm_forage[arm])

    per_rep: Dict[str, Any] = {}
    any_rep_clears_lift = False
    for rep in REPRESENTATIONS:
        off_arm, on_arm = _arm_id("off", rep), _arm_id("on", rep)
        off_s, on_s = _summ(off_arm), _summ(on_arm)
        off_mean = float(off_s["foraging_competence_mean"])
        on_mean = float(on_s["foraging_competence_mean"])
        on_per_seed = list(on_s["foraging_competence_per_seed"])
        # Strict majority of seeds where ON clears the lift-competence target.
        n_on_supra_target = int(sum(1 for v in on_per_seed if float(v) >= LIFT_COMPETENCE_TARGET))
        strict_majority = bool(n_on_supra_target > (len(on_per_seed) / 2.0)) if on_per_seed else False
        beats_off_by_margin = bool((on_mean - off_mean) >= LIFT_MARGIN)
        off_drift_flag = bool(off_mean >= LIFT_COMPETENCE_TARGET)  # OFF should NOT clear the target
        rep_clears = bool(strict_majority and beats_off_by_margin and not off_drift_flag)
        any_rep_clears_lift = any_rep_clears_lift or rep_clears
        per_rep[rep] = {
            "off_arm": off_arm, "on_arm": on_arm,
            "off_forage_mean": round(off_mean, 6), "off_forage_per_seed": off_s["foraging_competence_per_seed"],
            "on_forage_mean": round(on_mean, 6), "on_forage_per_seed": on_per_seed,
            "on_minus_off": round(on_mean - off_mean, 6),
            "on_clears_lift_target_strict_majority": strict_majority,
            "on_beats_off_by_lift_margin": beats_off_by_margin,
            "off_reproduces_rnd_plateau": bool(off_mean < LIFT_COMPETENCE_TARGET),
            "off_drift_flag": off_drift_flag,
            "rep_clears_lift_competent": rep_clears,
            "n_on_seeds_supra_lift_target": n_on_supra_target,
        }

    if not readiness_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
    elif any_rep_clears_lift:
        outcome, label = "PASS", "bootstrap_explorer_clears_lift_competent"
    else:
        outcome, label = "FAIL", "bootstrap_explorer_plateaus_capacity_gap_remains"

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
            {"name": "C_bootstrap_ON_clears_lift_competent_either_rep", "load_bearing": True,
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
            "any_rep_bootstrap_clears_lift_competent": bool(any_rep_clears_lift),
            "per_representation": per_rep,
            "lift_competence_target": LIFT_COMPETENCE_TARGET,
            "lift_margin": round(LIFT_MARGIN, 6),
            "d3_local_view_greedy_denominator": round(local_view_mean, 6),
            "d3_greedy_oracle": round(oracle_mean, 6),
            "d3_random_walk": round(_mean("random_walk"), 6),
            "reference_rnd_plateau_751": RND_PLATEAU,
            "reference_bc_expert_748": boot.BC_REFERENCE_32_72,
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
                "All OFF and ON boot arms emitted reuse-eligible per representation "
                "(rng_fully_reset via arm_cell + config_slice_declared + "
                "include_driver_script_in_hash=False). The mechanism logic lives in "
                "experiments/_lib/mech457_bootstrap_explorer + mech457_explorer_classes "
                "(in the substrate hash), so a later cross-driver MECH-457/INV-088 consumer "
                "-- esp. the V3-EXQ-750 strategy-diversity retest needing matched-competent "
                "policies on BOTH representations -- can cite reuse_baseline_from these cells."
            ),
        },
        "config": cfg,
        "load_bearing_dv": (
            "D3 foraging_competence (unshaped eval) under the composed bootstrap explorer (RND drive "
            "+ prioritized credit-replay + developmental coef/entropy anneal + increased budget), ON "
            "vs OFF (RND plateau), on BOTH z_world(cotrain) and raw 5x5. Verdict: ON clears the "
            "~13.05 lift-competence target (5.22 plateau + 7.83 margin) AND beats its OFF arm by >= "
            "the lift margin, on a strict majority of seeds, on EITHER representation; readiness = "
            "local_view_greedy + oracle clear the 1.0 floor @D3."
        ),
        "notes": (
            "MECH-457 competence bootstrap-explorer validation (post-build Step-8 retest). DIAGNOSTIC "
            "(excluded from scoring); PROMOTES/DEMOTES NOTHING; route to /failure-autopsy before any "
            "governance action. Substrate mech457_competence_bootstrap_explorer IMPLEMENTED 2026-07-16 "
            "(ree-v3 main 022a284). RE-DERIVE BRAKE RELEASED: >=2 non_contributory MECH-457 autopsies "
            "(751-750, 752-753-754, 755, 746c-756), but the named upstream substrate is now built AND "
            "this is a DISTINCT COMPOSED mechanism (new EXQ NUMBER, not a single-axis bolt-up). "
            "GOV-REUSE-1: the decisive readout (composed-bootstrap ON foraging vs 5.22 plateau on both "
            "reps) is in NO prior manifest (751 RND-only; 752-756 converters on the sparse base) -> "
            "run. NULL (ON plateaus on both reps) -> deeper capacity gap; route to /failure-autopsy "
            "(NOT another single-axis probe -- brake still holds for those). MECH-457 stays "
            "candidate/v3_pending; INV-088 candidate/pending_substrate_reconfirmation. OFF/ON minted "
            "reuse-eligible (mint-as-you-go)."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-765 MECH-457 competence bootstrap-explorer validation"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    if args.dry_run:
        seeds = list(fan.DRY_SEEDS)
        p0 = fan.DRY_P0
        on_budget = fan.DRY_RL
        off_budget = fan.DRY_RL
        eval_eps, steps = fan.DRY_EVAL, fan.DRY_STEPS
    else:
        seeds = list(fan.SEEDS)
        p0 = fan.P0_WARMUP_EPISODES
        off_budget = fan.RL_EPISODES                                  # 1000 -- the 751 plateau budget
        on_budget = int(fan.RL_EPISODES * boot.ON_BUDGET_MULTIPLIER)  # 3000 -- capacity to convert
        eval_eps, steps = fan.EVAL_EPISODES, fan.STEPS_PER_EPISODE

    result = run_experiment(seeds=seeds, p0=p0, on_budget=on_budget, off_budget=off_budget,
                            eval_eps=eval_eps, steps=steps)

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds, "rung": fan.RUNG_ID, "arms": list(ARM_ORDER),
        "representations": list(REPRESENTATIONS),
        "p0_warmup_episodes": p0, "on_budget_episodes": on_budget, "off_budget_episodes": off_budget,
        "on_budget_multiplier": boot.ON_BUDGET_MULTIPLIER,
        "eval_episodes": eval_eps, "steps_per_episode": steps,
        "actor_critic_hidden": fan.ACTOR_CRITIC_HIDDEN,
        "ac_lr": fan.AC_LR, "ac_gamma": fan.AC_GAMMA,
        "on_config": boot.make_on_config().as_slice(),
        "off_config": boot.make_off_config().as_slice(),
        "lift_competence_target": LIFT_COMPETENCE_TARGET, "lift_margin": round(LIFT_MARGIN, 6),
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
    for rep in REPRESENTATIONS:
        pr = hl["per_representation"][rep]
        print(
            f"  {rep}: OFF={pr['off_forage_mean']} ON={pr['on_forage_mean']} "
            f"(on-off={pr['on_minus_off']}; clears_lift={pr['rep_clears_lift_competent']})", flush=True,
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
