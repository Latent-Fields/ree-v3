#!/opt/local/bin/python3
"""V3-EXQ-882 -- MECH-472 HELD-OUT CONTEXT vs TASK MEMORISATION (EXP-0400).

WHY THIS EXISTS. MECH-471 is the generalisation claim that behavioural COMPETENCE updates
need the same bounded/provenanced/rollback-capable discipline already registered for memory
CONSOLIDATION (MECH-392/INV-080/MECH-401). If that discipline is ever built, something has to
decide WHEN a competence update has earned "durable" status. MECH-472 is the proposed
promotion gate: a competence update should be promoted to durable only on evidence from
CONTEXTS THAT DID NOT GENERATE IT. The registered FALSIFIER (claims.yaml) is a paired
in-context vs held-out-context evaluation of the SAME learned competence, swept across
training exposure: a genuine acquisition-vs-memorisation discriminator should show the gap
staying near zero while the competence is genuinely generalising, then widening once training
crosses into memorising idiosyncrasies of the specific contexts it was trained on
(overtraining). NOTE THE POLARITY: PASS (the gap widens with exposure, beyond a pre-registered
margin, while staying near zero at low exposure) SUPPORTS MECH-472 -- the gap is a usable
discriminator and therefore a usable promotion gate. FAIL (the gap never discriminates -- it
stays flat across the whole exposure sweep) WEAKENS MECH-472 -- the proposed gate would carry
no information, decision-relevant for whatever discipline MECH-471 might motivate building.

RELATIONSHIP TO EXP-0399 (V3-EXQ-875, MECH-471's own cheapest-first probe). The proposal notes
"reuse the EXP-0399 phase-1-trained baseline arm fingerprint if the competence set matches."
It does NOT match: EXP-0399 acquires TWO separable competences (reef geometry "corner" vs
"banded") on ONE shared agent to test CROSS-competence interference; this experiment trains
ONE competence and asks whether ITS OWN held-out-context evaluation discriminates acquisition
from memorisation. The "competence set" (what is trained, and why) differs, so the arm
fingerprint cannot match by construction (different config_slice, different training recipe
shape) and this experiment mints its own baseline from scratch. This is a genuine choice, not
an oversight: the reuse note in the proposal is conditional ("if the competence set matches")
and confirmed NOT to apply here. This experiment does NOT depend on EXP-0399/V3-EXQ-875 having
completed -- MECH-472's "lower priority than EXP-0399" note (claims.yaml, proposal notes) is a
governance PRIORITY ordering for the cluster, not a substrate precondition; nothing in this
design reads any V3-EXQ-875 output.

*** DO NOT CONFLATE WITH GOV-HELDOUT-1 *** -- that is the ASSEMBLY-side held-out validation of
governance workflow/rule changes, from the sibling intake of the same SkillOpt source
document. This claim is REE-INTERNAL and concerns the agent's own competence promotion. Per
GOV-ANALOGY-1 the resemblance is an analogy and is NEVER evidence in either direction.

WHAT "CONTEXT" MEANS HERE, AND WHY IT NEEDS NO NEW SUBSTRATE. CausalGridWorldV2 seeds ONE
`numpy.random.default_rng(seed)` at construction (`_rng`, causal_grid_world.py ~:1235) and
reset() draws from it WITHOUT re-seeding, so successive resets of one env instance produce a
sequence of DIFFERENT hazard/resource surface layouts (same task structure: same size, same
reef geometry, same n_reef_patches -- only the RNG-drawn point-hazard/resource positions
differ). A fresh env constructed with the SAME seed reproduces that EXACT sequence from its
start; a fresh env constructed with a DIFFERENT seed draws a DIFFERENT sequence of layouts
under the IDENTICAL task structure (same kwargs). So "the generating context set" = the
sequence of surface configurations a given seed's RNG stream produces, and "a held-out context
set matched on task structure but differing in surface configuration" = the SAME kwargs,
DIFFERENT seed. No context-set / domain-randomisation machinery needs to be built: the env's
own seeded RNG already IS that machinery, exactly as CLAUDE.md/queue-experiment's substrate
readiness check requires confirming before writing a script.

WHY SURVIVAL, NOT FORAGING (identical reasoning to V3-EXQ-875/EXP-0399). foraging_competence is
the metric the MECH-457 GOV-FANOUT-1 campaign (719a/724/728/734/742/780/788, 74% of that
campaign's autopsies routing to substrate_ceiling) has repeatedly found pinned near/at floor
for the all-ON stack. V3-EXQ-728 established the SAME all-ON stack DOES survive competently
even where it does not forage. So the competence trained and evaluated here is SURVIVAL/
AVOIDANCE under the "corner" reef geometry (3 fixed-corner reef safe-zone patches, legacy
_place_reef_patches, hazard_food_attraction=0.0 to decouple foraging from the DV) -- the same
env-kwargs variant "A" from V3-EXQ-875 EXP-0399, reused verbatim (not a new environment
configuration).

DESIGN. Paired in-context vs held-out-context evaluation of ONE trained competence, swept
across training EXPOSURE (total P1 REINFORCE episode count), per (seed x exposure) cell:
  Phase 0/1 (ACQUIRE). Fresh env + fresh agent per cell (full RNG reset, standard arm_cell
    convention). SD-056 e2 warmup (P0_WARMUP_EPISODES, ONCE per cell) then P1 REINFORCE (e2
    frozen, A0 recipe, exactly the V3-EXQ-875 / x734 `_train_all_on_agent` recipe) for
    `exposure` episodes, continuing on the SAME env instance (so P0+P1 together consume
    `P0_WARMUP_EPISODES + exposure` draws from that seed's RNG stream).
  Phase 2a (IN-CONTEXT EVAL). Evaluate on a FRESH env constructed with the SAME `seed` --
    this reproduces the START of that seed's RNG stream, i.e. episodes from the SAME
    generating process (config + seed) that produced the training experience.
  Phase 2b (HELD-OUT EVAL). Evaluate on a FRESH env constructed with a DIFFERENT seed
    (`seed + HELD_OUT_SEED_OFFSET`) but IDENTICAL kwargs -- same task structure, different
    surface configuration, never seen during training.
  PRIMARY DV: gap = in_context.survival_horizon - held_out.survival_horizon. Swept across
  `exposure` so the gap can be traced as a function of overtraining: a memorisation reading
  predicts the gap stays near zero at low exposure (genuine acquisition, no time to overfit
  idiosyncrasies of the training seed's specific draws) and WIDENS at high exposure
  (overtraining exploits training-context-specific structure that does not transfer).

ACCEPTANCE (mirrors EXP-0400 / claims.yaml verbatim, including the polarity):
  * PASS (outcome=PASS, evidence_direction=supports): the gap stays within the pre-registered
    noise margin at the LOWEST exposure level AND widens beyond that margin at the two highest
    exposure levels. Label: heldout_gap_discriminates_memorisation. The gap IS a usable
    acquisition-vs-memorisation discriminator.
  * FAIL (outcome=FAIL, evidence_direction=weakens): the gap stays within the noise margin
    across the FULL exposure sweep -- in-context and held-out performance move together.
    Label: heldout_gap_does_not_discriminate. Decision-relevant: MECH-472 would not be usable
    as the durability criterion for MECH-471.
  * Ambiguous dose-response (clears the margin only at the top, or non-monotonic beyond noise,
    or fails to stay near-zero at the lowest exposure while still widening at the top):
    outcome=FAIL, evidence_direction=mixed, label=heldout_gap_dose_response_ambiguous.
  * NON-DEGENERACY (outcome=FAIL, evidence_direction=unknown, non_degenerate=False,
    label=substrate_not_ready_requeue): EITHER the in-context trained competence fails to
    clear its survival floor (above the random-walk anchor by the declared margin) at the
    LOWEST exposure level on the worst seed (else there is no acquisition to test for), OR
    the HELD-OUT contexts are not genuinely reachable (random-walk survival_horizon on the
    held-out seed's env falls below a small reachability floor on any seed -- else a measured
    gap would be a FLOOR ARTEFACT, not evidence of memorisation, exactly the guard the
    proposal's acceptance_checks flags as "most likely to be skipped"), OR there is no live
    cross-seed variance in the acquired in-context competence.

evidence_direction_per_claim is NOT populated -- claim_ids has exactly one entry (MECH-472).

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

This module is ASCII-only in all runtime strings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.allon_training import _train_all_on_agent  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    RandomPolicy,
    REEForwardPolicy,
    evaluate_seed,
)
from experiments._lib.manifest_core import stamp_recording_core  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_882_mech472_context_memorization_generalization"
QUEUE_ID = "V3-EXQ-882"
CLAIM_IDS: List[str] = ["MECH-472"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Budget (real run) ----------------------------------------------------------------
SEEDS: List[int] = [42, 43, 45]   # 44 skipped per repo convention (recurring early-death
                                  # instability on this env family; substitute 45).
HELD_OUT_SEED_OFFSET = 10007      # deterministic per-seed held-out seed = seed + offset;
                                  # offset chosen far outside SEEDS/other-experiment ranges.
P0_WARMUP_EPISODES = 60           # SD-056 e2 forward-model warmup, ONCE per cell (A0 recipe:
                                  # e2 frozen through the following P1 exposure sweep).
EXPOSURE_LEVELS: List[int] = [50, 100, 250, 500]   # total P1 REINFORCE episodes -- the
                                                    # training-exposure sweep. [0] is the
                                                    # lowest (reference / near-zero-gap
                                                    # expectation), [-1] the highest
                                                    # (overtraining regime).
STEPS_PER_EPISODE = 150
EVAL_EPISODES = 12

# --- Dry-run budget ---------------------------------------------------------------------
DRY_SEEDS: List[int] = [42]
DRY_P0 = 2
DRY_EXPOSURE_LEVELS: List[int] = [2, 4]
DRY_STEPS = 15
DRY_EVAL = 2

# --- Pre-registered thresholds (declared as fractions of STEPS_PER_EPISODE; never derived
# from the run's own statistics) ----------------------------------------------------------
SURVIVAL_FLOOR_FRAC = 0.6          # in-context trained competence must survive >= 60% of the
                                   # episode at the lowest exposure level (acquisition floor).
RANDOM_MARGIN_FRAC = 0.15          # ... AND beat the random-walk anchor (same seed) by >= 15%
                                   # of the episode length (achievability / non-vacuity check).
REACHABLE_FLOOR_FRAC = 0.05        # held-out seed's RANDOM-WALK survival_horizon must clear
                                   # >= 5% of the episode length -- "genuinely reachable"
                                   # per the proposal's non-degeneracy guard; a near-zero
                                   # random-walk floor on the held-out layout would make any
                                   # measured gap a floor artefact, not memorisation evidence.
GAP_ABS_FLOOR_FRAC = 0.05          # absolute floor component of the gap-discrimination margin
GAP_SD_MULT = 2.0                  # scale-on-noise component (feedback_effect_size_pass_gate_
                                   # margin: scale on SD of the delta, plus an absolute floor),
                                   # noise SD estimated from the LOWEST exposure level's
                                   # per-seed gaps (the "near zero, not yet overtrained" cell).
CONFIRM_STRONG_FRAC = 0.5          # the two highest exposure levels must both clear >= this
                                   # fraction of the margin (guards against a single anomalous
                                   # top point reading as a confirmed widening).


def _env_kwargs() -> Dict[str, Any]:
    """The SAME "corner" reef-geometry survival/avoidance env as V3-EXQ-875 EXP-0399's
    variant "A" -- 3 reef safe-zone patches at fixed grid corners (legacy
    _place_reef_patches), scattered point hazards, hazard_food_attraction=0.0 so foraging is
    decoupled from the survival DV. Reused verbatim; not a new configuration. Only ONE
    competence is trained here (unlike EXP-0399's A/B pair), so only variant "A" is needed."""
    kw = dict(x724.ENV_KWARGS)
    kw["hazard_food_attraction"] = 0.0
    kw["reef_enabled"] = True
    kw["n_reef_patches"] = 3
    kw["reef_bipartite_layout"] = False   # legacy corner-pattern reef placement
    return kw


def _make_env(seed: int, env_kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The all-ON REE stack (724/734/742/875 recipe): the two-head REINFORCE bias substrate
    (lateral-PFC action bias + OFC devaluation bias) over the SD-056 e2 world-forward encoder.
    No new architecture; reused verbatim."""
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    cfg = REEConfig.from_dims(**kwargs)
    return REEAgent(cfg)


def _eval_agent(agent: REEAgent, env_kwargs: Dict[str, Any], seed: int, eval_eps: int,
                steps: int, policy_name: str) -> Dict[str, Any]:
    env = _make_env(seed, env_kwargs)
    return evaluate_seed(REEForwardPolicy(agent, name=policy_name), env, eval_eps, steps)


def _eval_random(env_kwargs: Dict[str, Any], seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    env = _make_env(seed, env_kwargs)
    return evaluate_seed(RandomPolicy(seed), env, eval_eps, steps)


def _held_out_seed(seed: int) -> int:
    return int(seed) + HELD_OUT_SEED_OFFSET


def _run_cell(
    seed: int, exposure: int, p0: int, eval_eps: int, steps: int,
    kw: Dict[str, Any], zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """One (seed, exposure) cell: acquire the SINGLE competence from scratch (fresh env +
    fresh agent, full RNG reset), then evaluate in-context (same seed) vs held-out (a
    different, never-trained-on seed under identical task structure)."""
    denom = p0 + max(exposure, 1)
    held_out = _held_out_seed(seed)

    env_train = _make_env(seed, kw)
    agent = _make_agent(env_train)
    _train_all_on_agent(
        agent, env_train, seed, p0_episodes=p0, p1_episodes=exposure,
        steps_per_episode=steps, rung_id=f"exposure_{exposure}",
        total_denominator=denom,
    )

    in_context = _eval_agent(agent, kw, seed, eval_eps, steps, "in_context")
    held_out_eval = _eval_agent(agent, kw, held_out, eval_eps, steps, "held_out")
    zg.observe(agent)

    gap = round(
        float(in_context["survival_horizon"]) - float(held_out_eval["survival_horizon"]), 6
    )
    return {
        "seed": int(seed),
        "held_out_seed": int(held_out),
        "exposure": int(exposure),
        "in_context": in_context,
        "held_out": held_out_eval,
        "gap_survival": gap,
    }


def run_experiment(
    seeds: List[int], exposures: List[int], p0: int, eval_eps: int, steps: int,
) -> Dict[str, Any]:
    kw = _env_kwargs()
    steps_f = float(steps)
    survival_floor = SURVIVAL_FLOOR_FRAC * steps_f
    random_margin = RANDOM_MARGIN_FRAC * steps_f
    reachable_floor = REACHABLE_FLOOR_FRAC * steps_f
    gap_abs_floor = GAP_ABS_FLOOR_FRAC * steps_f

    print(
        f"MECH-472 held-out context vs task memorisation ({len(exposures)} exposures x "
        f"{len(seeds)} seeds; P0={p0} steps={steps} eval={eval_eps})",
        flush=True,
    )

    # Random-walk anchors: cheap, computed once per seed -- readiness/achievability
    # reference only, not part of the boot training grid.
    random_in_context = {s: _eval_random(kw, s, eval_eps, steps) for s in seeds}
    random_held_out = {s: _eval_random(kw, _held_out_seed(s), eval_eps, steps) for s in seeds}

    zg = ZGoalStreamAccumulator()
    all_cells: List[Dict[str, Any]] = []
    per_exposure: Dict[int, List[Dict[str, Any]]] = {e: [] for e in exposures}

    for exposure in exposures:
        for seed in seeds:
            print(f"Seed {seed} Condition exposure_{exposure}", flush=True)
            slice_cfg = {
                "kind": "mech472_context_memorization_generalization",
                "env_kwargs": dict(kw),
                "p0_warmup_episodes": int(p0),
                "exposure_episodes": int(exposure),
                "steps_per_episode": int(steps),
                "eval_episodes": int(eval_eps),
                "held_out_seed_offset": int(HELD_OUT_SEED_OFFSET),
            }
            with arm_cell(
                seed, config_slice=slice_cfg, script_path=Path(__file__),
                config_slice_declared=True,
            ) as cell:
                row = _run_cell(seed, exposure, p0, eval_eps, steps, kw, zg)
                cell.stamp(row)
            all_cells.append(row)
            per_exposure[exposure].append(row)
            acquired_ok = (
                float(row["in_context"]["survival_horizon"]) >= survival_floor
            )
            print(
                f"verdict: {'PASS' if acquired_ok else 'FAIL'} "
                f"(exposure={exposure} seed={seed} "
                f"in_context={row['in_context']['survival_horizon']} "
                f"held_out={row['held_out']['survival_horizon']} "
                f"gap={row['gap_survival']})",
                flush=True,
            )

    control_rows = per_exposure[exposures[0]]   # lowest exposure level: the reference cell

    # ---- readiness / non-degeneracy ------------------------------------------------------
    in_context_worst = min(
        float(r["in_context"]["survival_horizon"]) for r in control_rows
    ) if control_rows else 0.0
    in_context_random_margin_worst = min(
        float(r["in_context"]["survival_horizon"])
        - float(random_in_context[r["seed"]]["survival_horizon"])
        for r in control_rows
    ) if control_rows else 0.0
    held_out_random_worst = min(
        float(random_held_out[s]["survival_horizon"]) for s in seeds
    ) if seeds else 0.0

    in_context_values = [float(r["in_context"]["survival_horizon"]) for r in control_rows]
    in_context_variance = (
        len(set(round(v, 3) for v in in_context_values)) > 1 if len(in_context_values) > 1 else False
    )

    acquisition_ok = bool(
        in_context_worst >= survival_floor
        and in_context_random_margin_worst >= random_margin
    )
    holdout_reachable_ok = bool(held_out_random_worst >= reachable_floor)
    cross_seed_variance_ok = bool(in_context_variance)

    # ---- dose-response readout ------------------------------------------------------------
    per_exposure_mean_gap: Dict[int, float] = {}
    per_exposure_sd_gap: Dict[int, float] = {}
    for e in exposures:
        gaps = [float(r["gap_survival"]) for r in per_exposure[e]]
        per_exposure_mean_gap[e] = round(statistics.fmean(gaps), 6) if gaps else 0.0
        per_exposure_sd_gap[e] = round(statistics.pstdev(gaps), 6) if len(gaps) > 1 else 0.0

    control_gaps = [float(r["gap_survival"]) for r in control_rows]
    control_sd = statistics.pstdev(control_gaps) if len(control_gaps) > 1 else 0.0
    gap_margin = round(max(gap_abs_floor, GAP_SD_MULT * control_sd), 6)

    lowest_exposure = exposures[0]
    max_exposure = exposures[-1]
    top_two = exposures[-2:] if len(exposures) >= 2 else exposures[-1:]
    top_two_gaps = [per_exposure_mean_gap[e] for e in top_two]

    near_zero_at_lowest = bool(abs(per_exposure_mean_gap[lowest_exposure]) <= gap_margin)
    gap_widens_confirmed = bool(
        near_zero_at_lowest
        and per_exposure_mean_gap[max_exposure] >= per_exposure_mean_gap[lowest_exposure] + gap_margin
        and all(g >= (CONFIRM_STRONG_FRAC * gap_margin) for g in top_two_gaps)
    )
    gap_flat = bool(
        all(abs(per_exposure_mean_gap[e]) <= gap_margin for e in exposures)
    )

    if not acquisition_ok or not holdout_reachable_ok or not cross_seed_variance_ok:
        outcome, direction, label = "FAIL", "unknown", "substrate_not_ready_requeue"
    elif gap_widens_confirmed:
        outcome, direction, label = "PASS", "supports", "heldout_gap_discriminates_memorisation"
    elif gap_flat:
        outcome, direction, label = "FAIL", "weakens", "heldout_gap_does_not_discriminate"
    else:
        outcome, direction, label = "FAIL", "mixed", "heldout_gap_dose_response_ambiguous"

    degeneracy = check_degeneracy({
        "in_context_survival_horizon": {"values": in_context_values, "floor": survival_floor},
        "held_out_random_walk_survival_horizon": {
            "values": [float(random_held_out[s]["survival_horizon"]) for s in seeds],
            "floor": reachable_floor,
        },
    })
    non_degenerate = bool(
        degeneracy["non_degenerate"] and acquisition_ok and holdout_reachable_ok
        and cross_seed_variance_ok
    )
    degeneracy_reason = degeneracy["degeneracy_reason"]
    if not acquisition_ok and not degeneracy_reason:
        degeneracy_reason = "in-context trained competence below floor or below random-walk margin at lowest exposure"
    if not holdout_reachable_ok and not degeneracy_reason:
        degeneracy_reason = "held-out context random-walk survival below reachability floor"
    if not cross_seed_variance_ok and not degeneracy_reason:
        degeneracy_reason = "no cross-seed variance in acquired in-context competence"

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "readiness": {
            "acquisition_ok": acquisition_ok,
            "holdout_reachable_ok": holdout_reachable_ok,
            "cross_seed_variance_ok": cross_seed_variance_ok,
            "in_context_worst_seed": round(in_context_worst, 6),
            "in_context_random_margin_worst_seed": round(in_context_random_margin_worst, 6),
            "held_out_random_walk_worst_seed": round(held_out_random_worst, 6),
            "survival_floor_ticks": round(survival_floor, 6),
            "random_margin_ticks": round(random_margin, 6),
            "reachable_floor_ticks": round(reachable_floor, 6),
        },
        "dose_response": {
            "per_exposure_mean_gap_survival": per_exposure_mean_gap,
            "per_exposure_sd_gap_survival": per_exposure_sd_gap,
            "gap_margin_ticks": gap_margin,
            "control_exposure_gap_sd": round(control_sd, 6),
            "near_zero_at_lowest_exposure": near_zero_at_lowest,
            "gap_widens_confirmed": gap_widens_confirmed,
            "gap_flat_across_sweep": gap_flat,
            "exposures": list(exposures),
        },
        "headline": {
            "lowest_exposure": lowest_exposure,
            "max_exposure": max_exposure,
            "mean_gap_at_lowest_exposure": per_exposure_mean_gap[lowest_exposure],
            "mean_gap_at_max_exposure": per_exposure_mean_gap[max_exposure],
        },
        "random_walk_anchors": {
            "in_context_survival_horizon_per_seed": {
                str(s): round(float(random_in_context[s]["survival_horizon"]), 6) for s in seeds
            },
            "held_out_survival_horizon_per_seed": {
                str(s): round(float(random_held_out[s]["survival_horizon"]), 6) for s in seeds
            },
        },
        "arm_results": all_cells,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "degenerate_metrics": degeneracy["degenerate_metrics"],
        "z_goal_stream_stats": zg.stats(),
    }


def _build_manifest(result: Dict[str, Any], timestamp_utc: str, dry_run: bool,
                    cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "queue_id": QUEUE_ID,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation_label": result["interpretation_label"],
        "readiness": result["readiness"],
        "dose_response": result["dose_response"],
        "headline": result["headline"],
        "random_walk_anchors": result["random_walk_anchors"],
        "arm_results": result["arm_results"],
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "degenerate_metrics": result["degenerate_metrics"],
        "sleep_driver_pattern": "none",
        "config": cfg,
        "load_bearing_dv": (
            "gap_survival = in-context survival_horizon (mean ticks survived/episode, "
            "evaluated on a fresh env constructed with the SAME seed used to train the "
            "competence) minus held-out survival_horizon (same statistic, evaluated on a "
            "fresh env constructed with a DIFFERENT seed, identical task-structure kwargs). "
            "Reported per training EXPOSURE level (total P1 REINFORCE episodes), swept from "
            "under-trained to overtrained. PASS (the gap stays near zero at the lowest "
            "exposure and widens beyond the pre-registered margin at the highest) SUPPORTS "
            "MECH-472; FAIL (the gap never discriminates across the whole sweep) WEAKENS it "
            "-- polarity is deliberate, see claims.yaml / EXP-0400 acceptance_checks."
        ),
        "notes": (
            "MECH-472 EXP-0400 held-out-context acquisition-vs-memorisation test. ONE "
            "competence (survival/avoidance under the V3-EXQ-875 EXP-0399 'corner' reef "
            "geometry, e2 frozen throughout per the 724-A0 recipe) trained per cell from "
            "scratch with a full RNG reset, then evaluated in-context (same training seed) "
            "vs held-out (different seed, identical env kwargs). No new context-set / "
            "domain-randomisation machinery: CausalGridWorldV2's own per-instance seeded RNG "
            "already generates a distinct hazard/resource surface configuration per seed under "
            "fixed task structure, which is exactly what 'generating context' vs 'held-out "
            "context' requires. Does not depend on V3-EXQ-875/EXP-0399 completing or on its "
            "output data -- the proposal's baseline-fingerprint-reuse note does not apply "
            "here (different competence-set/training-recipe shape), confirmed at authoring "
            "time; this experiment mints its own arms."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-882 MECH-472 held-out context vs task memorisation (EXP-0400)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--exposures", type=int, nargs="*", default=None)
    parser.add_argument("--p0", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    if args.dry_run:
        seeds = list(DRY_SEEDS)
        exposures = list(DRY_EXPOSURE_LEVELS)
        p0 = DRY_P0
        eval_eps, steps = DRY_EVAL, DRY_STEPS
    else:
        seeds = list(SEEDS)
        exposures = list(EXPOSURE_LEVELS)
        p0 = P0_WARMUP_EPISODES
        eval_eps, steps = EVAL_EPISODES, STEPS_PER_EPISODE

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.exposures is not None:
        exposures = [int(s) for s in args.exposures]
    if args.p0 is not None:
        p0 = int(args.p0)
    if args.eval_episodes is not None:
        eval_eps = int(args.eval_episodes)
    if args.steps is not None:
        steps = int(args.steps)

    if len(exposures) < 2:
        raise ValueError("exposures must contain at least 2 levels to trace a dose-response")
    exposures = sorted(exposures)

    result = run_experiment(
        seeds=seeds, exposures=exposures, p0=p0, eval_eps=eval_eps, steps=steps,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds,
        "exposures": exposures,
        "p0_warmup_episodes": p0,
        "eval_episodes": eval_eps,
        "steps_per_episode": steps,
        "env_kwargs": _env_kwargs(),
        "held_out_seed_offset": HELD_OUT_SEED_OFFSET,
        "survival_floor_frac": SURVIVAL_FLOOR_FRAC,
        "random_margin_frac": RANDOM_MARGIN_FRAC,
        "reachable_floor_frac": REACHABLE_FLOOR_FRAC,
        "gap_abs_floor_frac": GAP_ABS_FLOOR_FRAC,
        "gap_sd_mult": GAP_SD_MULT,
        "confirm_strong_frac": CONFIRM_STRONG_FRAC,
    }
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run), cfg=cfg)
    stamp_recording_core(
        manifest, config=cfg, seeds=seeds, script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )

    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    )
    out_path = write_flat_manifest(
        manifest, out_dir, dry_run=args.dry_run, config=cfg, seeds=seeds,
        script_path=Path(__file__),
        elapsed_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        stamp=False,  # already stamped above (need z_goal_stream_stats folded in first)
        z_goal_stream_stats=result.get("z_goal_stream_stats"),
    )

    print(f"manifest: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} label={result['interpretation_label']} "
        f"direction={result['evidence_direction']} "
        f"acquisition_ok={result['readiness']['acquisition_ok']} "
        f"holdout_reachable_ok={result['readiness']['holdout_reachable_ok']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(
        f"  per_exposure_mean_gap_survival="
        f"{result['dose_response']['per_exposure_mean_gap_survival']}",
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
