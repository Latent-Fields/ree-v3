#!/opt/local/bin/python3
"""V3-EXQ-882a -- MECH-472 HELD-OUT CONTEXT vs TASK MEMORISATION (EXP-0400).
(supersedes V3-EXQ-882 -- z_world warmup wired + fraction-basis acquisition gate.)

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

=====================================================================================
WHAT V3-EXQ-882a CORRECTS (two independent defects, both confirmed in
failure_autopsy_V3-EXQ-882_2026-08-03, Sections 7, 9 and 10).
=====================================================================================

DEFECT 1 -- UNTRAINED z_world (CONFIRMED, Section 9/10; the load-bearing fix). V3-EXQ-882's
single `_train_all_on_agent(...)` acquisition call NEVER passed `zworld_p0_episodes`, so it
defaulted to 0 -- the exact SD-070 defect documented in ree-v3/CLAUDE.md ("SD-070 ADOPTION in
the _train_all_on_agent driver family"): with no P0a warmup, `split_encoder.world_encoder` is
never stepped and z_world stays a FROZEN RANDOM PROJECTION for the whole run, silently (no
error). This is the identical defect confirmed twice elsewhere in this driver family
(V3-EXQ-728 original "3/3 seeds failed", and V3-EXQ-875/MECH-471 -- fixed by
V3-EXQ-875a the same day). A frozen random z_world is precisely the kind of per-seed-arbitrary
noise source that produces a STABLE, SEED-DEPENDENT (not exposure-dependent) split -- one
seed's random projection happens to support the policy well enough to clear the floor and stay
solved, others' do not, and NO amount of additional REINFORCE exposure fixes that. That is
exactly the shape 882 observed (1/3 seeds solved immediately and stayed solved; 2/3 never
approached the floor across a 10x exposure range with no improving trend), and it is
indistinguishable in 882's own evidence from a genuine hazard-geometry difficulty split.
  FIX: wire `zworld_p0_episodes=ZWORLD_P0_EPISODES=60` (SD-070's V3-EXQ-728/875a validated
  operating point) into the acquisition call, with a DEDICATED warmup env (same seed + kwargs;
  the warmup consumes env RNG on its own instance so it does NOT shift the layout sequence the
  e2-warmup/P1 exposure sweep then sees on env_train). This is a strict improvement regardless
  of which reading turns out right: either it clears the per-seed split (informative), or it
  does not (which STRENGTHENS the genuine-difficulty reading by ruling this confound out). Per
  autopsy Section 10 it is a HARD PRECONDITION on any 882 successor, not a judgement call.

DEFECT 2 -- FRAGILE WORST-OF-3-SEED ACQUISITION GATE (Section 5 cluster pattern; Section 7
point 1). 882's acquisition-floor readiness precondition was aggregated WORST-OF-N-SEED
(`min(...)` over 3 seeds at the lowest exposure). A single hard/unlucky seed's natural
(non-error) outcome therefore zeroed the ENTIRE run's read -- even though the seeds that DID
clear are themselves fully informative. This conflates STOCHASTIC readiness (does a given
seed's environment-difficulty draw permit acquisition at low exposure? -- a genuinely random
per-seed event, correct for SOME seeds not to clear) with a hard pass/fail veto on the run's
own load-bearing adjudication.
  FIX (mirrors the V3-EXQ-873a fraction-gate fix): raise seed count 3 -> 8 for power on the
  fraction statistic, and adjudicate the acquisition floor on a FRACTION BASIS -- the gate is
  green when the fraction of seeds clearing the acquisition floor at the lowest exposure
  EXCEEDS ACQUISITION_SEED_FRACTION_FLOOR (majority; measured = n_cleared / n_seeds, a floor).
  Seeds that do NOT clear are recorded INFORMATIONALLY (per_seed_acquisition_difficulty block)
  and EXCLUDED from the load-bearing gap dose-response cohort (a floor-artefact gap from a
  non-acquiring seed is exactly what MECH-472's own non-degeneracy guard warns against), never
  counted as a hard failure. The gap DV is computed over the CLEARED cohort only.

DEFECT 3 (recording gap, Section 7 point 3) -- 882 recorded only the worst-seed readiness
summary, forcing the autopsy to reconstruct the per-seed difficulty split from `arm_results`
by hand. 882a emits an explicit top-level `per_seed_acquisition_difficulty` block: per seed,
the in-context / held-out / gap survival trajectory across every exposure, the random anchor,
whether it cleared the acquisition floor at the lowest exposure, and whether it ever clears.

Do NOT simply extend exposure further on the same recipe: 882's dose-response was flat across
a 10x exposure range (50 -> 500) on the failing seeds with no improving trend, which already
argues against "more of the same training alone" fixing them (autopsy Section 7 point 2). The
correct levers are the two defects above, not a power-bump of the braked design.

RELATIONSHIP TO EXP-0399 (V3-EXQ-875, MECH-471's own cheapest-first probe). The proposal notes
"reuse the EXP-0399 phase-1-trained baseline arm fingerprint if the competence set matches."
It does NOT match: EXP-0399 acquires TWO separable competences (reef geometry "corner" vs
"banded") on ONE shared agent to test CROSS-competence interference; this experiment trains
ONE competence and asks whether ITS OWN held-out-context evaluation discriminates acquisition
from memorisation. The "competence set" (what is trained, and why) differs, so the arm
fingerprint cannot match by construction (different config_slice, different training recipe
shape) and this experiment mints its own baseline from scratch. This is a genuine choice, not
an oversight. This experiment does NOT depend on EXP-0399/V3-EXQ-875 having completed --
nothing in this design reads any V3-EXQ-875 output.

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
own seeded RNG already IS that machinery.

WHY SURVIVAL, NOT FORAGING (identical reasoning to V3-EXQ-875/EXP-0399). foraging_competence is
the metric the MECH-457 GOV-FANOUT-1 campaign has repeatedly found pinned near/at floor for
the all-ON stack. V3-EXQ-728 established the SAME all-ON stack DOES survive competently even
where it does not forage. So the competence trained and evaluated here is SURVIVAL/AVOIDANCE
under the "corner" reef geometry (3 fixed-corner reef safe-zone patches, legacy
_place_reef_patches, hazard_food_attraction=0.0 to decouple foraging from the DV) -- the same
env-kwargs variant "A" from V3-EXQ-875 EXP-0399, reused verbatim (not a new configuration).

DESIGN. Paired in-context vs held-out-context evaluation of ONE trained competence, swept
across training EXPOSURE (total P1 REINFORCE episode count), per (seed x exposure) cell:
  Phase 0a (z_world WARMUP -- THE 882a FIX). SD-070 z_world encoder warmup for
    ZWORLD_P0_EPISODES episodes on a DEDICATED env (same seed + kwargs), so z_world is a
    TRAINED encoder, not a frozen random projection, before any competence is acquired.
  Phase 0b/1 (ACQUIRE). Fresh env + fresh agent per cell (full RNG reset, standard arm_cell
    convention). SD-056 e2 warmup (P0_WARMUP_EPISODES, ONCE per cell) then P1 REINFORCE (e2
    frozen, A0 recipe, exactly the V3-EXQ-875a / x734 `_train_all_on_agent` recipe) for
    `exposure` episodes, continuing on the SAME env instance.
  Phase 2a (IN-CONTEXT EVAL). Evaluate on a FRESH env constructed with the SAME `seed` --
    episodes from the SAME generating process (config + seed) that produced the training
    experience.
  Phase 2b (HELD-OUT EVAL). Evaluate on a FRESH env constructed with a DIFFERENT seed
    (`seed + HELD_OUT_SEED_OFFSET`) but IDENTICAL kwargs -- same task structure, different
    surface configuration, never seen during training.
  PRIMARY DV: gap = in_context.survival_horizon - held_out.survival_horizon, computed over the
  CLEARED cohort (seeds that cleared the acquisition floor at the lowest exposure) and swept
  across `exposure`: a memorisation reading predicts the gap stays near zero at low exposure
  and WIDENS at high exposure (overtraining).

ACCEPTANCE (mirrors EXP-0400 / claims.yaml verbatim, including the polarity):
  * PASS (outcome=PASS, evidence_direction=supports): the cohort-mean gap stays within the
    pre-registered noise margin at the LOWEST exposure level AND widens beyond that margin at
    the two highest exposure levels. Label: heldout_gap_discriminates_memorisation.
  * FAIL (outcome=FAIL, evidence_direction=weakens): the gap stays within the noise margin
    across the FULL exposure sweep -- in-context and held-out performance move together.
    Label: heldout_gap_does_not_discriminate.
  * Ambiguous dose-response: outcome=FAIL, evidence_direction=mixed,
    label=heldout_gap_dose_response_ambiguous.
  * NON-DEGENERACY (outcome=FAIL, evidence_direction=unknown, non_degenerate=False,
    label=substrate_not_ready_requeue): EITHER the fraction of seeds clearing the acquisition
    floor (above the random-walk anchor by the declared margin) at the LOWEST exposure level
    does NOT exceed ACQUISITION_SEED_FRACTION_FLOOR (else there is no acquisition to test for),
    OR the HELD-OUT contexts of the cleared cohort are not genuinely reachable (random-walk
    survival on a cleared seed's held-out env below the reachability floor -- else a measured
    gap would be a FLOOR ARTEFACT), OR there is no live cross-seed variance in the acquired
    in-context competence of the cleared cohort.

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

EXPERIMENT_TYPE = "v3_exq_882a_mech472_context_memorization_generalization"
QUEUE_ID = "V3-EXQ-882a"
SUPERSEDES = "v3_exq_882_mech472_context_memorization_generalization"
CLAIM_IDS: List[str] = ["MECH-472"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Budget (real run) ----------------------------------------------------------------
# Seed count raised 3 -> 8 for adequate power on the acquisition-clear FRACTION statistic
# (882's base rate was ~1/3; a fraction gate needs enough seeds that a majority read is
# robust to a couple of hard environment-difficulty draws). Continuity with 882's seeds
# (42, 43, 45) is preserved so the two runs can be compared per-seed. Seed 44 skipped per
# repo convention (recurring early-death instability on this env family).
SEEDS: List[int] = [42, 43, 45, 46, 47, 48, 49, 50]
HELD_OUT_SEED_OFFSET = 10007      # deterministic per-seed held-out seed = seed + offset;
                                  # offset chosen far outside SEEDS/other-experiment ranges.
ZWORLD_P0_EPISODES = 60           # 882a FIX (DEFECT 1): SD-070 z_world ENCODER warmup, run
                                  # AHEAD of the e2 warmup on a DEDICATED env per cell.
                                  # 60 = the V3-EXQ-728/875a validated SD-070 operating point.
P0_WARMUP_EPISODES = 60           # SD-056 e2 forward-model warmup, ONCE per cell (A0 recipe:
                                  # e2 frozen through the following P1 exposure sweep).
EXPOSURE_LEVELS: List[int] = [50, 100, 250, 500]   # total P1 REINFORCE episodes -- the
                                                    # training-exposure sweep. [0] lowest
                                                    # (near-zero-gap expectation), [-1] highest
                                                    # (overtraining regime).
STEPS_PER_EPISODE = 150
EVAL_EPISODES = 12

# --- Dry-run budget ---------------------------------------------------------------------
DRY_SEEDS: List[int] = [42, 43]
DRY_ZWORLD_P0 = 2
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
                                   # per the proposal's non-degeneracy guard.
GAP_ABS_FLOOR_FRAC = 0.05          # absolute floor component of the gap-discrimination margin
GAP_SD_MULT = 2.0                  # scale-on-noise component (feedback_effect_size_pass_gate_
                                   # margin: scale on SD of the delta, plus an absolute floor),
                                   # noise SD estimated from the LOWEST exposure level's cohort
                                   # per-seed gaps.
CONFIRM_STRONG_FRAC = 0.5          # the two highest exposure levels must both clear >= this
                                   # fraction of the margin (guards against a single anomalous
                                   # top point reading as a confirmed widening).
# 882a FIX (DEFECT 2): fraction-basis acquisition gate. The acquisition floor is adjudicable
# when the FRACTION of seeds clearing it at the lowest exposure EXCEEDS this floor (majority),
# NOT worst-of-N. measured = n_cleared / n_seeds; a floor, indexer-recomputable in spirit.
ACQUISITION_SEED_FRACTION_FLOOR = 0.5   # majority: with 8 seeds, need >= 5 of 8 to clear.


def _env_kwargs() -> Dict[str, Any]:
    """The SAME "corner" reef-geometry survival/avoidance env as V3-EXQ-875 EXP-0399's
    variant "A" -- 3 reef safe-zone patches at fixed grid corners (legacy
    _place_reef_patches), scattered point hazards, hazard_food_attraction=0.0 so foraging is
    decoupled from the survival DV. Reused verbatim; not a new configuration."""
    kw = dict(x724.ENV_KWARGS)
    kw["hazard_food_attraction"] = 0.0
    kw["reef_enabled"] = True
    kw["n_reef_patches"] = 3
    kw["reef_bipartite_layout"] = False   # legacy corner-pattern reef placement
    return kw


def _make_env(seed: int, env_kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **env_kwargs)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """The all-ON REE stack (724/734/742/875a recipe): the two-head REINFORCE bias substrate
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
    seed: int, exposure: int, p0: int, zworld_p0: int, eval_eps: int, steps: int,
    kw: Dict[str, Any], zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """One (seed, exposure) cell: acquire the SINGLE competence from scratch (fresh env +
    fresh agent, full RNG reset), then evaluate in-context (same seed) vs held-out (a
    different, never-trained-on seed under identical task structure).

    882a FIX: the acquisition call now runs the SD-070 z_world P0a warmup first
    (`zworld_p0_episodes=zworld_p0`) on a DEDICATED warmup env (same seed + kwargs), so
    z_world is a trained encoder rather than a frozen random projection."""
    denom = p0 + max(exposure, 1)
    held_out = _held_out_seed(seed)

    env_train = _make_env(seed, kw)
    agent = _make_agent(env_train)
    _train_all_on_agent(
        agent, env_train, seed, p0_episodes=p0, p1_episodes=exposure,
        steps_per_episode=steps, rung_id=f"exposure_{exposure}",
        total_denominator=denom,
        zworld_p0_episodes=zworld_p0,
        # DEDICATED env for the warmup: the P0a rollout consumes env RNG, so reusing
        # env_train would shift the layout sequence P0b/P1 then see. Same seed + kwargs.
        zworld_p0_env=(_make_env(seed, kw) if zworld_p0 > 0 else None),
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
    seeds: List[int], exposures: List[int], p0: int, zworld_p0: int, eval_eps: int, steps: int,
) -> Dict[str, Any]:
    kw = _env_kwargs()
    steps_f = float(steps)
    survival_floor = SURVIVAL_FLOOR_FRAC * steps_f
    random_margin = RANDOM_MARGIN_FRAC * steps_f
    reachable_floor = REACHABLE_FLOOR_FRAC * steps_f
    gap_abs_floor = GAP_ABS_FLOOR_FRAC * steps_f

    print(
        f"MECH-472 held-out context vs task memorisation ({len(exposures)} exposures x "
        f"{len(seeds)} seeds; zworld_p0={zworld_p0} P0={p0} steps={steps} eval={eval_eps})",
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
                "zworld_p0_episodes": int(zworld_p0),
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
                row = _run_cell(seed, exposure, p0, zworld_p0, eval_eps, steps, kw, zg)
                cell.stamp(row)
            all_cells.append(row)
            per_exposure[exposure].append(row)
            acquired_ok_cell = (
                float(row["in_context"]["survival_horizon"]) >= survival_floor
            )
            print(
                f"verdict: {'PASS' if acquired_ok_cell else 'FAIL'} "
                f"(exposure={exposure} seed={seed} "
                f"in_context={row['in_context']['survival_horizon']} "
                f"held_out={row['held_out']['survival_horizon']} "
                f"gap={row['gap_survival']})",
                flush=True,
            )

    control_rows = per_exposure[exposures[0]]   # lowest exposure level: the reference cell
    lowest_exposure = exposures[0]
    max_exposure = exposures[-1]

    # ---- per-seed acquisition clearance at the LOWEST exposure (FRACTION basis, 882a FIX) ----
    def _cleared(row: Dict[str, Any]) -> bool:
        ic = float(row["in_context"]["survival_horizon"])
        margin = ic - float(random_in_context[row["seed"]]["survival_horizon"])
        return bool(ic >= survival_floor and margin >= random_margin)

    per_seed_cleared: Dict[int, bool] = {int(r["seed"]): _cleared(r) for r in control_rows}
    cleared_seeds: List[int] = [s for s in seeds if per_seed_cleared.get(int(s), False)]
    not_cleared_seeds: List[int] = [s for s in seeds if not per_seed_cleared.get(int(s), False)]
    n_seeds = len(seeds)
    n_cleared = len(cleared_seeds)
    clear_fraction = (n_cleared / n_seeds) if n_seeds else 0.0
    acquisition_ok = bool(clear_fraction > ACQUISITION_SEED_FRACTION_FLOOR)

    cleared_set = set(int(s) for s in cleared_seeds)
    # Reachability + variance are assessed over the CLEARED cohort (the only seeds contributing
    # to the load-bearing gap). Falls back to all seeds only to compute a reportable worst when
    # the cohort is empty (acquisition_ok is already False in that case).
    reach_cohort = cleared_seeds if cleared_seeds else list(seeds)
    held_out_random_worst = min(
        float(random_held_out[s]["survival_horizon"]) for s in reach_cohort
    ) if reach_cohort else 0.0
    holdout_reachable_ok = bool(cleared_seeds and held_out_random_worst >= reachable_floor)

    cohort_ic_values = [
        float(r["in_context"]["survival_horizon"])
        for r in control_rows if int(r["seed"]) in cleared_set
    ]
    cross_seed_variance_ok = bool(
        len(set(round(v, 3) for v in cohort_ic_values)) > 1
        if len(cohort_ic_values) > 1 else False
    )

    # ---- dose-response readout (CLEARED cohort only for the load-bearing gap) ---------------
    per_exposure_mean_gap: Dict[int, float] = {}
    per_exposure_sd_gap: Dict[int, float] = {}
    per_exposure_mean_gap_all_seeds: Dict[int, float] = {}
    per_exposure_cohort_n: Dict[int, int] = {}
    for e in exposures:
        cohort_gaps = [
            float(r["gap_survival"]) for r in per_exposure[e] if int(r["seed"]) in cleared_set
        ]
        all_gaps = [float(r["gap_survival"]) for r in per_exposure[e]]
        per_exposure_mean_gap[e] = round(statistics.fmean(cohort_gaps), 6) if cohort_gaps else 0.0
        per_exposure_sd_gap[e] = round(statistics.pstdev(cohort_gaps), 6) if len(cohort_gaps) > 1 else 0.0
        per_exposure_mean_gap_all_seeds[e] = round(statistics.fmean(all_gaps), 6) if all_gaps else 0.0
        per_exposure_cohort_n[e] = len(cohort_gaps)

    control_gaps = [
        float(r["gap_survival"]) for r in control_rows if int(r["seed"]) in cleared_set
    ]
    control_sd = statistics.pstdev(control_gaps) if len(control_gaps) > 1 else 0.0
    gap_margin = round(max(gap_abs_floor, GAP_SD_MULT * control_sd), 6)

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
        "in_context_survival_horizon_cohort": {
            "values": cohort_ic_values, "floor": survival_floor,
        },
        "held_out_random_walk_survival_horizon_cohort": {
            "values": [float(random_held_out[s]["survival_horizon"]) for s in reach_cohort],
            "floor": reachable_floor,
        },
    })
    non_degenerate = bool(
        degeneracy["non_degenerate"] and acquisition_ok and holdout_reachable_ok
        and cross_seed_variance_ok
    )
    degeneracy_reason = degeneracy["degeneracy_reason"]
    if not acquisition_ok and not degeneracy_reason:
        degeneracy_reason = (
            "acquisition-floor clear fraction %.3f did not exceed floor %.3f at lowest exposure "
            "(cleared %d/%d seeds)" % (clear_fraction, ACQUISITION_SEED_FRACTION_FLOOR, n_cleared, n_seeds)
        )
    if not holdout_reachable_ok and not degeneracy_reason:
        degeneracy_reason = "held-out context random-walk survival below reachability floor on the cleared cohort"
    if not cross_seed_variance_ok and not degeneracy_reason:
        degeneracy_reason = "no cross-seed variance in acquired in-context competence of the cleared cohort"

    # ---- per-seed acquisition difficulty (882a FIX, DEFECT 3: explicit, not reconstructed) --
    per_seed_acq: Dict[str, Any] = {}
    for s in seeds:
        ic_by_e: Dict[str, float] = {}
        ho_by_e: Dict[str, float] = {}
        gap_by_e: Dict[str, float] = {}
        for e in exposures:
            for r in per_exposure[e]:
                if int(r["seed"]) == int(s):
                    ic_by_e[str(e)] = round(float(r["in_context"]["survival_horizon"]), 4)
                    ho_by_e[str(e)] = round(float(r["held_out"]["survival_horizon"]), 4)
                    gap_by_e[str(e)] = round(float(r["gap_survival"]), 4)
        rand_ic = round(float(random_in_context[s]["survival_horizon"]), 4)
        ic_lowest = ic_by_e.get(str(lowest_exposure), 0.0)
        per_seed_acq[str(s)] = {
            "in_context_survival_by_exposure": ic_by_e,
            "held_out_survival_by_exposure": ho_by_e,
            "gap_survival_by_exposure": gap_by_e,
            "random_in_context_survival": rand_ic,
            "random_margin_at_lowest": round(ic_lowest - rand_ic, 4),
            "cleared_acquisition_floor_at_lowest": bool(per_seed_cleared.get(int(s), False)),
            "ever_clears_floor": bool(any(v >= survival_floor for v in ic_by_e.values())),
        }

    return {
        "outcome": outcome,
        "evidence_direction": direction,
        "interpretation_label": label,
        "readiness": {
            "acquisition_ok": acquisition_ok,
            "acquisition_basis": "seed_fraction_majority",
            "acquisition_clear_fraction": round(clear_fraction, 4),
            "acquisition_seed_fraction_floor": ACQUISITION_SEED_FRACTION_FLOOR,
            "n_seeds": int(n_seeds),
            "n_cleared_at_lowest": int(n_cleared),
            "cleared_seeds": sorted(int(s) for s in cleared_seeds),
            "not_cleared_seeds": sorted(int(s) for s in not_cleared_seeds),
            "holdout_reachable_ok": holdout_reachable_ok,
            "cross_seed_variance_ok": cross_seed_variance_ok,
            "held_out_random_walk_worst_cohort": round(held_out_random_worst, 6),
            "gap_cohort": "cleared_seeds_only",
            "survival_floor_ticks": round(survival_floor, 6),
            "random_margin_ticks": round(random_margin, 6),
            "reachable_floor_ticks": round(reachable_floor, 6),
        },
        "dose_response": {
            "per_exposure_mean_gap_survival": per_exposure_mean_gap,
            "per_exposure_sd_gap_survival": per_exposure_sd_gap,
            "per_exposure_mean_gap_survival_all_seeds": per_exposure_mean_gap_all_seeds,
            "per_exposure_cohort_n": per_exposure_cohort_n,
            "gap_margin_ticks": gap_margin,
            "control_exposure_gap_sd": round(control_sd, 6),
            "near_zero_at_lowest_exposure": near_zero_at_lowest,
            "gap_widens_confirmed": gap_widens_confirmed,
            "gap_flat_across_sweep": gap_flat,
            "exposures": list(exposures),
        },
        "per_seed_acquisition_difficulty": per_seed_acq,
        "headline": {
            "lowest_exposure": lowest_exposure,
            "max_exposure": max_exposure,
            "mean_gap_at_lowest_exposure_cohort": per_exposure_mean_gap[lowest_exposure],
            "mean_gap_at_max_exposure_cohort": per_exposure_mean_gap[max_exposure],
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
        "supersedes": SUPERSEDES,
        "timestamp_utc": timestamp_utc,
        "dry_run": bool(dry_run),
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "interpretation_label": result["interpretation_label"],
        "readiness": result["readiness"],
        "dose_response": result["dose_response"],
        "per_seed_acquisition_difficulty": result["per_seed_acquisition_difficulty"],
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
            "fresh env constructed with a DIFFERENT seed, identical task-structure kwargs), "
            "computed over the CLEARED cohort (seeds that cleared the acquisition floor at the "
            "lowest exposure) and reported per training EXPOSURE level. PASS (the cohort gap "
            "stays near zero at the lowest exposure and widens beyond the pre-registered margin "
            "at the highest) SUPPORTS MECH-472; FAIL (the gap never discriminates across the "
            "whole sweep) WEAKENS it -- polarity is deliberate, see claims.yaml / EXP-0400."
        ),
        "notes": (
            "V3-EXQ-882a supersedes V3-EXQ-882 (MECH-472 EXP-0400 held-out-context "
            "acquisition-vs-memorisation test). Two confirmed 882 defects corrected "
            "(failure_autopsy_V3-EXQ-882_2026-08-03 Sections 7/9/10): DEFECT 1 -- 882's "
            "acquisition call never passed zworld_p0_episodes, so the SD-070 z_world warmup "
            "never ran and z_world stayed a frozen random projection all run (the exact "
            "V3-EXQ-728/875a/MECH-471 defect); 882a wires zworld_p0_episodes=60 on a dedicated "
            "warmup env. DEFECT 2 -- 882's worst-of-3-seed acquisition-floor veto was too "
            "fragile; 882a raises seed count 3->8 and adjudicates the floor on a fraction "
            "basis (majority of seeds must clear; non-clearing seeds informational and excluded "
            "from the load-bearing gap cohort), mirroring the V3-EXQ-873a fix. DEFECT 3 -- "
            "882a emits an explicit per_seed_acquisition_difficulty block. Same env/recipe "
            "otherwise (survival/avoidance under the V3-EXQ-875 EXP-0399 'corner' reef "
            "geometry, e2 frozen throughout per the 724-A0 recipe). Mints its own arms; does "
            "not depend on V3-EXQ-875 output."
        ),
    }


def main() -> Tuple[Optional[str], Optional[str], bool]:
    parser = argparse.ArgumentParser(
        description="V3-EXQ-882a MECH-472 held-out context vs task memorisation (EXP-0400)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--exposures", type=int, nargs="*", default=None)
    parser.add_argument("--zworld-p0", type=int, default=None)
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
        zworld_p0 = DRY_ZWORLD_P0
        p0 = DRY_P0
        eval_eps, steps = DRY_EVAL, DRY_STEPS
    else:
        seeds = list(SEEDS)
        exposures = list(EXPOSURE_LEVELS)
        zworld_p0 = ZWORLD_P0_EPISODES
        p0 = P0_WARMUP_EPISODES
        eval_eps, steps = EVAL_EPISODES, STEPS_PER_EPISODE

    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    if args.exposures is not None:
        exposures = [int(s) for s in args.exposures]
    if args.zworld_p0 is not None:
        zworld_p0 = int(args.zworld_p0)
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
        seeds=seeds, exposures=exposures, p0=p0, zworld_p0=zworld_p0,
        eval_eps=eval_eps, steps=steps,
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = {
        "seeds": seeds,
        "exposures": exposures,
        "zworld_p0_episodes": zworld_p0,
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
        "acquisition_seed_fraction_floor": ACQUISITION_SEED_FRACTION_FLOOR,
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
        f"clear_fraction={result['readiness']['acquisition_clear_fraction']} "
        f"holdout_reachable_ok={result['readiness']['holdout_reachable_ok']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(
        f"  per_exposure_mean_gap_survival (cohort)="
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
