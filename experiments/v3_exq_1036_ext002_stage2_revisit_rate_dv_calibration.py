#!/opt/local/bin/python3
"""
V3-EXQ-1036 -- EXT-002 lineage stage 2a: revisit-rate DV calibration
(residue-frozen ablation, draw-to-MIN_POOLED_SEEDS, NO pre-registered verdict)

WHY THIS RUN EXISTS, AND WHY IT IS NOT "STAGE 2" AS RATIFIED
---------------------------------------------------------------------------
V3-EXQ-983a's confirmed diagnostic (`failure_autopsy_V3-EXQ-1014_2026-09-09.md`,
Step 8 gate 2026-09-09T00:55:13Z) ratified an EXT-002 lineage stage 2: "residue-
frozen ablation with a hold-aware revisit-AVOIDANCE DV (fresh-decision revisits
to erred keys per unit exposure, early vs late, A0 minus A1) ... MIN_POOLED_SEEDS
3, seeds drawn until the floor is met."

A subsequent READ-ONLY feasibility pass
(`REE_assembly/evidence/planning/ext002_stage2_dv_threshold_feasibility_20260911.md`,
origin/master dc63fe68eb) found that NO numeric pass/fail threshold for that DV
is derivable from any banked manifest -- the statistic (a per-unit-exposure
REVISIT RATE) has never been computed by any driver script; 983a's own `decline`
field is a different, BARRED statistic (a conditional repeat-OUTCOME rate, not a
revisit-RATE). The feasibility doc's recommendation (option a): "run a short,
cheap calibration pass ... whose only purpose is to bank the new DV on a handful
of seeds and set the bar from that" rather than committing the full run to an
ungrounded pre-registered bar.

THIS SCRIPT implements that calibration pass, with one deliberate adaptation
from the feasibility doc's literal "6-8 seeds" framing, stated so a later reader
does not read it as an oversight: it reuses stage 2's OWN ratified draw target
(MIN_POOLED_SEEDS = 3), not a separate 6-8-seed population. Rationale: a
calibration-only draw to 6-8 survivors would cost roughly DOUBLE the already-
priced stage-2 compute (survivor cells dominate cost; the survival rate is the
same ~1/8), which is not "short, cheap" in absolute terms. Reusing the SAME
draw-to-3 design costs the SAME order of magnitude as the previously-priced full
run (~27-38h, ~12-24 draws) but removes the one thing that made committing to
that run unwise: this run pre-registers NO C1-equivalent threshold and emits NO
verdict on EXT-002/ARC-013. It purely BANKS the DV's real, freshly-drawn range
(and the revisit-VOLUME aggregation comparison the feasibility doc's point (b)
asks for), so a follow-on /governance or /queue-experiment cycle can set stage
2's real threshold from actual data instead of a proxy computed over n=2 (one of
which is a 0/0 degenerate). If 3 pooled seeds prove too thin a range once this
lands, extending the draw (a higher target on a follow-up EXQ) is cheap relative
to re-deriving a threshold from nothing.

CLAIM-FREE, DELIBERATELY (governance instruction): tags no claim_ids, carries
1014's `bears_on` tokens verbatim (`actor_adequacy_monostrategy` --
NOT a registry qid, see 1014's autopsy Section 2 -- and
`residue_error_persistence_readout`). This is diagnostic/calibration
infrastructure for the lineage, not a claim test.

SUBSTRATE REUSE (deliberate, not a copy-paste): every substrate-facing helper
below is IMPORTED from V3-EXQ-983a's already-authored, already-red-teamed
module (`_make_env`, `_make_agent`, `run_cell`, `_action_stream_divergence`,
`_positive_control_e1_prediction_error`, the training-completion-gate bands,
`arm_contexts`/`ARM_INTACT`/`ARM_FROZEN`). Only the SEED-DRAW loop and the DV
AGGREGATION/ANALYSIS are new. This is the same reuse pattern already used
elsewhere in this tree (e.g. V3-EXQ-1023 imports four sibling experiment
modules); it minimises new surface area for a run whose entire point is
methodological caution. `full_config` below is kept BYTE-IDENTICAL to 983a's
own (same env/agent/training config) so per-cell `arm_fingerprint`s remain
comparable across the two scripts (no overlap is expected in practice, since
this run's seeds are drawn to explicitly EXCLUDE 983/983a/1014's 8 pinned
seeds -- see EXCLUDED_SEEDS -- but the design keeps the door open rather than
closing it for no reason).

THE NEW DV, PRECISELY
---------------------------------------------------------------------------
983a's `run_cell` already records, per (seed, arm) cell: `n_revisits_early`,
`n_revisits_late` (counts of FRESH-DECISION revisits to a previously-erred
(cell,action) key -- already gated on `ticks["e3_tick"]`, i.e. already
"hold-aware": a HELD/latched tick can never contribute a revisit event, exactly
the routing note's requirement) and `realized_total_steps`. Stage 2's DV needs
only a normalisation this driver adds:

    half = max(1, realized_total_steps // 2)          # matches 983a's own split
    rate_early = n_revisits_early / half
    rate_late  = n_revisits_late  / half
    decline_rate = rate_early - rate_late              # per (seed, arm)
    decline_rate_gap = decline_rate_A0 - decline_rate_A1   # per seed, A0 minus A1

This is a RATE (revisits per unit exposure), not 983a's conditional repeat-
OUTCOME probability -- the two are explicitly different statistics built from
the same underlying event log (feasibility doc Section 2).

WHAT THIS RUN DOES NOT DO (by design, and each is a stated deferral, not an
omission):
  - No PASS/FAIL verdict on EXT-002 or ARC-013 (claim-free; no criterion is
    registered against `decline_rate_gap`).
  - No P7-style "DV freedom under a control policy" certification and no
    action-stream-divergence / action-class-diversity GATES. The routing note's
    full apparatus is for the EVIDENCE run; this is calibration. Divergence and
    diversity ARE measured and recorded per pooled seed (reusing 983a's own
    `_action_stream_divergence`) as non-gating diagnostics, since they cost
    nothing extra to record.
  - No decision on whether stage 2's eventual revisit-volume precondition
    should use a min-across-pooled-cells gate (983a's structure, defeated by a
    single seed-31-like draw) or a more robust median/majority aggregation
    (feasibility doc point (b)). This run RECORDS both statistics
    (`min_across_pooled_cells_revisits_early`,
    `median_across_pooled_cells_revisits_early`) over whatever it draws, so the
    decision can be made from real data, but does not make the decision itself
    -- that is a design call for the eventual evidence-run's authoring session,
    owed to /governance per the feasibility doc.

NON-DEGENERACY (breach -> requeue label, NOT a verdict on EXT-002/ARC-013)
---------------------------------------------------------------------------
  pooled_seed_count            >= MIN_USABLE_POOLED_SEEDS (2) -- below this
                                there is no population to characterise a range
                                over at all.
  revisit_population_non_degenerate  at least one pooled seed has a non-zero
                                TOTAL REVISIT COUNT in BOTH arms (i.e. the draw
                                did not land entirely on seed-31-like "constant
                                mover" profiles, which the feasibility doc
                                measured as a real, non-hypothetical risk: 1 of
                                983a's 2 completion-gate survivors was exactly
                                this profile). Judged on REVISIT counts, not on
                                distinct-erred-key counts -- red-team F2
                                (Step 4.5, opus, 2026-09-14): a seed can carry
                                erred keys yet never revisit one, which a
                                keys-based test would misread as non-degenerate
                                while the DV is still a structural 0/0 on that
                                seed.

SLEEP: not used (no sleep flags set) -- no SLEEP DRIVER line required.

RED-TEAM (Step 4.5, opus, 2026-09-14): CONTESTED -- seven findings across the
four causal-chain families (manipulation->DV, criterion discrimination, verdict
grid, self-certifying gate). Every claim was checked against source (983a's
line numbers, or 983a's own real-scale banked manifest) before acting:

  F1 (CONFIRMED BY ARITHMETIC, most serious) -- revisit-degenerate pooled seeds
     (zero total revisits in either arm) were counted into the banked
     decline_rate_gap range/mean, diluting it toward zero exactly as the
     feasibility doc found for the n=2 proxy this run exists to replace.
     FIXED: `per_seed_rate_dv` now tags `revisit_degenerate`; the aggregate
     statistics are computed over non-degenerate pooled seeds only, with the
     degenerate ones still recorded verbatim in `per_seed_dv`.
  F2 (CONFIRMED BY MEASUREMENT) -- the `revisit_population_non_degenerate`
     precondition tested `n_distinct_erred_keys`, a different quantity than
     the DV itself (`n_revisits_*`); a seed can carry erred keys with zero
     revisits and pass a keys-based test while still being a structural 0/0.
     FIXED: degeneracy is now judged on total revisit counts directly (same
     fix underlies F1).
  F3 (CONFIRMED BY MEASUREMENT) -- the `criteria` block's threshold was an
     integer (2.0) compared strictly against an integer `n_pooled`, the exact
     boundary-disagreement shape 983a's own red-team already fixed once
     (F4-B) elsewhere in this lineage; separately, `passed` reflected only
     population size, so the `all_degenerate` FAIL branch could carry a
     `passed: true` load-bearing criterion. FIXED: half-integer threshold
     (matching the preconditions), `passed` now requires both population
     adequacy and non-degeneracy.
  F4 (CONFIRMED BY READING) -- a readout key named
     `control_arm_A0_decline_rate_mean` mislabelled A0 (the INTACT/treatment
     arm) as the control arm (A1 is `x983a.DV_HEADROOM_CONTROL_ARM`), inviting
     a future reader deriving a bar to lift the wrong arm's value. FIXED:
     renamed to `treatment_arm_A0_decline_rate_mean`.
  F5 (ACCEPTED, disclosed rather than gated) -- this run's completion gate
     omits 983a's own evidence-run P4 revisit-denominator floor, so it pools
     more liberally than a P4-gated evidence run would and may bank a
     lower-revisit-volume population. Not fixed by adding a new pre-registered
     floor (that would reintroduce the ungrounded-threshold problem this run
     exists to avoid): `per_seed_dv[].clears_legacy_p4_floor` records it per
     seed, and the banked-outcome interpretation text says so explicitly.
  F6 (CONFIRMED BY READING) -- the draw loop reimplemented the completion-gate
     band check inline, silently dropping 983a's `exclusion_asymmetry`
     one-armed-exclusion report (983a's own Family-4 red-team requirement).
     FIXED: the loop now calls `x983a.completion_gate(rows_all, seeds_drawn)`
     directly each iteration; the manifest's `completion_gate` block is that
     function's own return value verbatim.
  F7 (DISMISSED, upstream) -- the per-cell `arm_fingerprint` cannot distinguish
     A0 from A1 (`config_slice` carries no arm discriminator), confirmed on
     983a's own real-scale manifest (byte-identical A0/A1 fingerprint blocks
     for the same seed). This is 983a's design, inherited unchanged, not a
     defect this driver introduces -- disposition belongs with that module,
     not here (Phase 0 arm-fingerprinting is emit-only and never serves a
     cached cell, so it cannot silently collapse this run's own arms).

Every finding fixed or dismissed in writing above with a source citation. One
pass only, per the skill's "do NOT iterate to CLEAR" rule -- none of F1-F7
changed the DV formula, the manipulation, or introduced a new criterion; all
are aggregation/reporting fixes internal to the calibration bookkeeping, so no
second spawn was made.
"""

import argparse
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.manifest_core import stamp_recording_core

import experiments.v3_exq_983a_ext002_residue_error_persistence_headroom as x983a

EXPERIMENT_TYPE = "v3_exq_1036_ext002_stage2_revisit_rate_dv_calibration"
CLAIM_IDS: List[str] = []  # claim-free -- deliberate, see module docstring
EXPERIMENT_PURPOSE = "baseline"

# ARM_FINGERPRINT_EXEMPT: every cell's RNG reset + arm_fingerprint is already
# discharged by x983a.run_cell()'s own `arm_cell()` context manager (that
# module: reset_all_rng(seed) on cell entry, cell.stamp(row) on exit -- verified
# by reading the source; every row this driver receives already carries
# row["arm_fingerprint"]). This driver never builds a cell itself, it only
# calls x983a.run_cell(), so validate_experiments.py's AST scan -- which
# inspects only THIS file's own source and cannot see across the module
# boundary into an imported function -- reports it missing. Exempted rather
# than duplicating (and risking a double RNG reset from) logic that already
# runs correctly one call frame away.
ARM_FINGERPRINT_EXEMPT = (
    "reset_all_rng()+arm_fingerprint discharged inside the imported "
    "x983a.run_cell()'s arm_cell() context manager; this driver only calls "
    "that function and never builds a cell directly -- see comment above"
)
BEARS_ON = ["actor_adequacy_monostrategy", "residue_error_persistence_readout"]
LINEAGE_NOTE = (
    "V3-EXQ-983 / V3-EXQ-983a / V3-EXQ-1014 lineage. Stage 2 per the 1014 autopsy's "
    "confirmed routing_note (2026-09-09T00:55:13Z); this run is the calibration-first "
    "alternative recommended by "
    "REE_assembly/evidence/planning/ext002_stage2_dv_threshold_feasibility_20260911.md "
    "(option a), NOT stage 2 as ratified -- see module docstring for the target-size "
    "adaptation."
)

# --- draw design ------------------------------------------------------------
# Reuses stage 2's OWN ratified floor (983a routing_note) rather than a separate
# 6-8-seed target -- see module docstring "THIS SCRIPT implements..." for why.
MIN_POOLED_SEEDS = 3
MIN_USABLE_POOLED_SEEDS = 2  # below this the calibration is uninformative
MAX_DRAWS = 30  # ratified pricing implies ~12-24 draws at ~1/8 survival; margin above that

# Freshly-drawn seeds must be independent of the lineage's already-used seeds
# (983 / 983a / 1014's 8 pinned boards, plus 983a's own P3 positive-control probe
# seed) -- the feasibility doc's entire point is that a calibration basis drawn
# from the SAME 8 seeds is not independent of the population stage 2 will pool.
EXCLUDED_SEEDS = {42, 123, 456, 7, 11, 17, 23, 31, x983a.PROBE_SEED}
SEED_DRAW_RNG_SEED = 20260909  # deterministic; recorded in the manifest
SEED_DRAW_LOW, SEED_DRAW_HIGH = 1000, 999_999

# Training regime IDENTICAL to 983a's validated one (not 1014's shorter one) --
# this calibration must characterise the DV under the SAME budget the eventual
# evidence run will use, and 983a's own docstring notes survival is invariant to
# episode count (dying seeds die in ~16 steps of every episode regardless).
WARMUP_EPISODES = 110
STEPS_PER_EPISODE = 200

def _full_config(warmup_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    """BYTE-IDENTICAL to 983a's own `full_config` (env/agent/training constants) so
    per-cell `arm_fingerprint`s stay comparable and the substrate is provably the
    same one 983a/1014 validated. Only `warmup_episodes`/`steps_per_episode` are
    parameterised (both default to 983a's own values above)."""
    return {
        "env": "CausalGridWorldV2",
        "size": 6,
        "num_hazards": 4,
        "num_resources": 3,
        "hazard_harm": 0.02,
        "proximity_harm_scale": 0.05,
        "use_proxy_fields": True,
        "self_dim": 32,
        "world_dim": 32,
        "alpha_world": 0.9,
        "alpha_self": 0.3,
        "lr": 1e-3,
        "num_candidates": x983a.NUM_CANDIDATES,
        "warmup_episodes": warmup_episodes,
        "steps_per_episode": steps_per_episode,
        "unified_latent_mode": False,
        "reafference_action_dim": 0,
        "arms": x983a.ARMS,
        "selection_rule": "argmin_over_candidate_residue_scores",
        "repeat_error_key": "(pre_action_grid_x, pre_action_grid_y, executed_action_class)",
        "control_probe_steps": x983a.CONTROL_PROBE_STEPS,
    }


def _draw_fresh_seed(rng: random.Random, already_drawn: set) -> int:
    while True:
        s = rng.randint(SEED_DRAW_LOW, SEED_DRAW_HIGH)
        if s not in EXCLUDED_SEEDS and s not in already_drawn:
            return s


def _rates(row: Dict[str, Any]) -> Dict[str, float]:
    half = max(1, int(row["realized_total_steps"]) // 2)
    rate_early = row["n_revisits_early"] / half
    rate_late = row["n_revisits_late"] / half
    return {
        "rate_early": float(rate_early),
        "rate_late": float(rate_late),
        "decline_rate": float(rate_early - rate_late),
    }


def per_seed_rate_dv(row_a0: Dict[str, Any], row_a1: Dict[str, Any]) -> Dict[str, Any]:
    """The per-unit-exposure revisit-avoidance DV for one pooled seed pair.

    See module docstring "THE NEW DV, PRECISELY" for the derivation. Both input
    rows come straight from `x983a.run_cell`'s output -- no new substrate call.

    RED-TEAM (Step 4.5, opus, 2026-09-14) F1/F2 FIX: degeneracy is judged on
    REVISIT COUNTS (`n_revisits_early + n_revisits_late`), not on
    `n_distinct_erred_keys` -- a seed can have erred keys but zero revisits
    (never re-encountering the (cell,action) pair), which still yields a
    structurally-0/half=0 rate on that arm and would silently pass a
    keys-based degeneracy test while still diluting the banked range toward
    zero (F1's exact finding). `revisit_degenerate` is True if EITHER arm's
    total revisit count is 0.
    """
    a0 = _rates(row_a0)
    a1 = _rates(row_a1)
    a0_total_revisits = int(row_a0["n_revisits_early"]) + int(row_a0["n_revisits_late"])
    a1_total_revisits = int(row_a1["n_revisits_early"]) + int(row_a1["n_revisits_late"])
    return {
        "seed": int(row_a0["seed"]),
        "A0_rate_early": a0["rate_early"],
        "A0_rate_late": a0["rate_late"],
        "A0_decline_rate": a0["decline_rate"],
        "A1_rate_early": a1["rate_early"],
        "A1_rate_late": a1["rate_late"],
        "A1_decline_rate": a1["decline_rate"],
        "decline_rate_gap": a0["decline_rate"] - a1["decline_rate"],
        "n_distinct_erred_keys_A0": int(row_a0["n_distinct_erred_keys"]),
        "n_distinct_erred_keys_A1": int(row_a1["n_distinct_erred_keys"]),
        "n_total_revisits_A0": a0_total_revisits,
        "n_total_revisits_A1": a1_total_revisits,
        "revisit_degenerate": bool(a0_total_revisits == 0 or a1_total_revisits == 0),
        # Informational only (F5) -- 983a's OWN evidence-run P4 floor
        # (early-half revisits, both arms >= 4.5) is NOT inherited as a gate
        # here (this run pools more liberally so it can characterise what a
        # freshly-drawn population actually looks like); flagging which
        # pooled seeds would/would not have cleared it lets a reader see that
        # the banked range may include lower-revisit-volume seeds than a
        # P4-gated evidence run would pool.
        "clears_legacy_p4_floor": bool(
            row_a0["n_revisits_early"] > x983a.FLOOR_REVISIT_DENOMINATOR_P4
            and row_a1["n_revisits_early"] > x983a.FLOOR_REVISIT_DENOMINATOR_P4
        ),
    }


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else float("nan")


def _range(vals: List[float]) -> float:
    return float(max(vals) - min(vals)) if vals else float("nan")


def _sd(vals: List[float]) -> float:
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def run(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()

    warmup = min(3, WARMUP_EPISODES) if dry_run else WARMUP_EPISODES
    steps = min(20, STEPS_PER_EPISODE) if dry_run else STEPS_PER_EPISODE
    full_config = _full_config(warmup, steps)

    target = min(2, MIN_POOLED_SEEDS) if dry_run else MIN_POOLED_SEEDS
    max_draws = min(4, MAX_DRAWS) if dry_run else MAX_DRAWS

    # P3 positive control -- global, reused from 983a verbatim, isolated from
    # every scored cell below. Not gated here (informational; the substrate's
    # error-signal non-degeneracy was already confirmed by 983a/800).
    pe_probe = x983a._positive_control_e1_prediction_error(full_config, dry_run)
    print(
        f"[V3-EXQ-1036] P3 positive control: e1_prediction_error_min="
        f"{pe_probe['e1_prediction_error_min']:.6g} "
        f"(n_harm_events={pe_probe['n_harm_events_observed']})",
        flush=True,
    )

    rng = random.Random(SEED_DRAW_RNG_SEED)
    seeds_drawn: List[int] = []
    pooled_seeds: List[int] = []
    rows_all: List[Dict[str, Any]] = []
    agents: List[Any] = []
    divergence_by_seed: Dict[str, Any] = {}
    diversity_by_seed: Dict[str, Any] = {}
    completion_detail: Dict[str, Any] = {}

    # RED-TEAM (Step 4.5, opus, 2026-09-14) F6 FIX: reuse x983a.completion_gate
    # directly (rather than a hand-rolled per-seed band check) so this driver
    # inherits its `exclusion_asymmetry` / one-armed-exclusion reporting for
    # free -- 983a's own red-team required that block (Family 4) and a
    # reimplementation silently dropped it.
    n_draws = 0
    while len(pooled_seeds) < target and n_draws < max_draws:
        seed = _draw_fresh_seed(rng, set(seeds_drawn))
        seeds_drawn.append(seed)
        n_draws += 1

        seed_episode_actions: Dict[str, List[List[int]]] = {}
        for ctx in x983a.arm_contexts():
            row, agent, episode_actions = x983a.run_cell(
                arm_ctx=ctx,
                seed=seed,
                full_config=full_config,
                warmup_episodes=warmup,
                steps_per_episode=steps,
                dry_run=dry_run,
            )
            rows_all.append(row)
            agents.append(agent)
            seed_episode_actions[ctx["id"]] = episode_actions

        completion_detail = x983a.completion_gate(rows_all, seeds_drawn)
        pooled_seeds = completion_detail["pooled_seeds"]
        newly_pooled = seed in pooled_seeds
        if newly_pooled:
            div = x983a._action_stream_divergence(
                seed_episode_actions[x983a.ARM_INTACT],
                seed_episode_actions[x983a.ARM_FROZEN],
            )
            divergence_by_seed[str(seed)] = div
            diversity_by_seed[str(seed)] = {
                a: len({ac for ep in seed_episode_actions[a] for ac in ep})
                for a in x983a.ARMS
            }
        print(
            f"[V3-EXQ-1036] draw {n_draws}/{max_draws}: seed={seed} "
            f"pooled={newly_pooled} pooled_so_far={len(pooled_seeds)}/{target}",
            flush=True,
        )

    # ---- calibration DV over pooled seeds -----------------------------------
    per_seed_dv: List[Dict[str, Any]] = []
    for s in pooled_seeds:
        row_a0 = next(
            r for r in rows_all if r["seed"] == s and r["arm_id"] == x983a.ARM_INTACT
        )
        row_a1 = next(
            r for r in rows_all if r["seed"] == s and r["arm_id"] == x983a.ARM_FROZEN
        )
        per_seed_dv.append(per_seed_rate_dv(row_a0, row_a1))

    # RED-TEAM (Step 4.5, opus, 2026-09-14) F1 FIX: the banked range/mean/sd is
    # computed over NON-DEGENERATE pooled seeds only. Including a seed whose
    # DV is a structural 0/0 (or 0/half=0) dilutes the range toward zero by
    # construction -- exactly the defect measured on 983a's own real pooled
    # population {456, 31} (range 0.00102, half the mass from seed 31's exact
    # zero) that this run exists to characterise honestly, not reproduce
    # silently. Degenerate seeds are STILL recorded in full in `per_seed_dv`
    # (never dropped from the manifest), just excluded from the aggregate.
    nondegenerate_dv = [d for d in per_seed_dv if not d["revisit_degenerate"]]
    decline_rate_gaps = [d["decline_rate_gap"] for d in nondegenerate_dv]
    a0_declines = [d["A0_decline_rate"] for d in nondegenerate_dv]
    a1_declines = [d["A1_decline_rate"] for d in nondegenerate_dv]  # control-arm range

    n_pooled = len(pooled_seeds)
    n_degenerate = sum(1 for d in per_seed_dv if d["revisit_degenerate"])
    n_nondegenerate = n_pooled - n_degenerate
    all_degenerate = n_pooled > 0 and n_degenerate == n_pooled
    pooled_ok = n_pooled >= MIN_USABLE_POOLED_SEEDS
    non_degenerate = bool(pooled_ok and not all_degenerate)
    thin_nondegenerate_range = bool(non_degenerate and n_nondegenerate < 2)

    control_arm_range = _range(a1_declines)
    dv_range_gap = _range(decline_rate_gaps)

    # Revisit-volume aggregation comparison (feasibility doc point (b)) -- over
    # EARLY-half counts of every pooled cell, both arms (983a's own P4 scope).
    pooled_set = set(pooled_seeds)
    pooled_cell_revisits_early = [
        int(r["n_revisits_early"]) for r in rows_all if int(r["seed"]) in pooled_set
    ]
    min_across_pooled_cells = (
        min(pooled_cell_revisits_early) if pooled_cell_revisits_early else 0
    )
    median_across_pooled_cells = (
        statistics.median(pooled_cell_revisits_early)
        if pooled_cell_revisits_early
        else 0.0
    )

    if not pooled_ok:
        outcome = "FAIL"
        label = "stage2_dv_calibration_insufficient_population"
        direction = "diagnostic_no_direction"
        text = (
            f"CALIBRATION INCOMPLETE: only {n_pooled}/{target} target seeds cleared "
            f"the inherited training-completion gate within the {max_draws}-draw "
            f"budget ({n_draws} drawn). This itself is evidence for the revisit-"
            "volume risk flagged by "
            "ext002_stage2_dv_threshold_feasibility_20260911.md: freshly-drawn "
            "seeds do not reliably survive stage 2's inherited completion gate. Not "
            "a verdict on EXT-002/ARC-013 (claim-free). Route: /governance should "
            "raise MAX_DRAWS and/or revisit the ~1/8 survival-rate assumption "
            "before re-attempting."
        )
    elif all_degenerate:
        outcome = "FAIL"
        label = "stage2_dv_calibration_all_seeds_revisit_degenerate"
        direction = "diagnostic_no_direction"
        text = (
            "CALIBRATION DEGENERATE: every pooled seed had zero total revisits "
            "in at least one arm (the seed-31 zero-revisit profile), so the "
            "per-unit-exposure DV is 0/0 for every pooled seed and no usable range "
            "can be read off this run. This reproduces, at a larger n, the exact "
            "risk the feasibility doc quantified from n=2. Route: /governance "
            "decides whether to raise MIN_POOLED_SEEDS, screen degenerate seeds at "
            "draw time, or accept the individual per_seed_dv values banked here."
        )
    else:
        outcome = "PASS"
        label = "stage2_dv_calibration_banked"
        direction = "diagnostic_no_direction"
        _thin_note = (
            f" CAUTION: only {n_nondegenerate} non-degenerate pooled seed(s) "
            "contributed to this range -- read it as illustrative, not a "
            "calibrated bar, until more seeds are drawn."
            if thin_nondegenerate_range
            else ""
        )
        text = (
            f"CALIBRATION BANKED: {n_pooled} freshly-drawn seeds ({n_draws} draws) "
            f"cleared the inherited training-completion gate, of which "
            f"{n_nondegenerate} were revisit-non-degenerate (both arms >0 total "
            f"revisits) and used for the range below; {n_degenerate} were "
            "revisit-degenerate and are recorded in per_seed_dv but excluded from "
            f"the aggregate.{_thin_note} decline_rate_gap (A0 minus A1) over "
            f"non-degenerate pooled seeds: mean={_mean(decline_rate_gaps):.6g}, "
            f"range={dv_range_gap:.6g}. A1 control-arm decline_rate range="
            f"{control_arm_range:.6g} -- the statistic "
            "ext002_stage2_dv_threshold_feasibility_20260911.md Section 2 "
            "recommends deriving a future C1-equivalent bar from. NOT a verdict on "
            "EXT-002/ARC-013 -- claim-free, no pre-registered threshold. This "
            "pool was NOT gated on 983a's own evidence-run P4 revisit-denominator "
            "floor (see per_seed_dv[].clears_legacy_p4_floor), so it may include "
            "lower-revisit-volume seeds than a P4-gated evidence run would admit "
            "-- read the banked range with that in mind. "
            "Governance/queue-experiment should set stage 2's evidence-run "
            "threshold from this range, and resolve the feasibility doc's point "
            f"(b) -- revisit-volume aggregation: min-across-pooled-cells="
            f"{min_across_pooled_cells} vs median-across-pooled-cells="
            f"{median_across_pooled_cells:.1f} over the early-half counts recorded "
            "here -- before queuing the full evidence run."
        )

    # RED-TEAM (Step 4.5, opus, 2026-09-14) F3 FIX: half-integer threshold
    # (matching the preconditions below and 983a's own F4-B convention) so an
    # integer-valued `n_pooled` sitting exactly on the floor cannot read MET
    # under one convention and UNMET under another. `passed` now reflects the
    # FULL adequacy bar (population size AND not-all-degenerate), so a FAIL
    # outcome can never carry a `passed: true` load-bearing criterion.
    criteria_passed = bool(pooled_ok and not all_degenerate)
    criteria = [
        {
            "name": "calibration_population_adequate",
            "load_bearing": True,
            "passed": criteria_passed,
            "measured": float(n_pooled),
            "threshold": float(MIN_USABLE_POOLED_SEEDS) - 0.5,
        },
    ]
    criteria_non_degenerate = {"calibration_population_adequate": non_degenerate}

    preconditions = [
        {
            "name": "pooled_seed_count",
            "description": (
                "number of freshly-drawn seeds clearing the inherited "
                "training-completion gate (steps_realized_frac>=0.60, "
                "harm_rate_train<=0.35, both arms)"
            ),
            "control": "same completion-gate bands 983a/1014 inherited",
            "measured": float(n_pooled),
            "threshold": float(MIN_USABLE_POOLED_SEEDS) - 0.5,
            "direction": "lower",
            "met": bool(pooled_ok),
        },
        {
            "name": "revisit_population_non_degenerate",
            "description": (
                "at least one pooled seed has a NON-ZERO TOTAL REVISIT COUNT "
                "(n_revisits_early + n_revisits_late) in BOTH arms -- not just a "
                "non-zero distinct-erred-key count, which a seed can carry while "
                "still generating zero revisits (red-team F2: erred keys and "
                "revisits are different quantities; a keys-based test would pass "
                "a seed whose DV is nonetheless a structural 0/0)"
            ),
            "control": "pooled seeds surviving the completion gate",
            "measured": float(n_pooled - n_degenerate) if n_pooled else 0.0,
            "threshold": 0.5,
            "direction": "lower",
            "met": bool(not all_degenerate) if n_pooled > 0 else False,
        },
    ]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    readout = {
        "n_draws": int(n_draws),
        "n_pooled": int(n_pooled),
        "target_pooled": int(target),
        "decline_rate_gap_mean": _mean(decline_rate_gaps),
        "decline_rate_gap_range": dv_range_gap,
        "decline_rate_gap_sd": _sd(decline_rate_gaps),
        "control_arm_A1_decline_rate_range": control_arm_range,
        "control_arm_A1_decline_rate_mean": _mean(a1_declines),
        # RED-TEAM (Step 4.5, opus, 2026-09-14) F4 FIX: A0 is the INTACT
        # (treatment) arm, not the control arm -- see x983a.DV_HEADROOM_CONTROL_ARM
        # = "A1_RESIDUE_FROZEN". The old key name ("control_arm_A0_...") invited a
        # future reader to lift the wrong arm's value when deriving a C1-equivalent
        # bar from `control_arm_A1_decline_rate_range` above.
        "treatment_arm_A0_decline_rate_mean": _mean(a0_declines),
        "min_across_pooled_cells_revisits_early": float(min_across_pooled_cells),
        "median_across_pooled_cells_revisits_early": float(median_across_pooled_cells),
        "n_revisit_degenerate_pooled_seeds": int(n_degenerate),
        "n_nondegenerate_pooled_seeds": int(n_nondegenerate),
        "thin_nondegenerate_range": int(thin_nondegenerate_range),
        "pooled_ok": int(pooled_ok),
        "non_degenerate_flag": int(non_degenerate),
    }

    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "lineage_note": LINEAGE_NOTE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": "" if non_degenerate else text,
        "positive_control_e1_prediction_error": pe_probe,
        "seed_draw": {
            "rng_seed": SEED_DRAW_RNG_SEED,
            "excluded_seeds": sorted(EXCLUDED_SEEDS),
            "seeds_drawn": seeds_drawn,
            "pooled_seeds": pooled_seeds,
            "n_draws": n_draws,
            "max_draws": max_draws,
            "target_pooled": target,
        },
        # x983a.completion_gate()'s own return, called against the full
        # accumulated (rows_all, seeds_drawn) -- carries `exclusion_asymmetry`
        # (983a's Family-4 one-armed-exclusion report) for free (F6 fix).
        "completion_gate": completion_detail,
        "per_seed_dv": per_seed_dv,
        "action_stream_divergence_by_seed": divergence_by_seed,
        "action_class_diversity_by_seed": diversity_by_seed,
        "revisit_volume_aggregation": {
            "min_across_pooled_cells_revisits_early": min_across_pooled_cells,
            "median_across_pooled_cells_revisits_early": median_across_pooled_cells,
            "note": (
                "recorded for the /governance decision on whether stage 2's "
                "eventual revisit-volume precondition uses a min-across-pooled-"
                "cells gate (983a's structure) or a more robust median/majority "
                "aggregation -- feasibility doc point (b). This run does not "
                "decide it."
            ),
        },
        "interpretation": {
            "label": label,
            "text": text,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": criteria,
        "readout": readout,
        "arm_results": rows_all,
    }

    stamp_recording_core(
        manifest,
        config=full_config,
        seeds=seeds_drawn if seeds_drawn else [0],
        script_path=Path(__file__),
        started_at=t0,
        agent=agents,
    )

    print("\n[V3-EXQ-1036] Results", flush=True)
    print(
        f"  n_draws={n_draws} n_pooled={n_pooled}/{target} "
        f"decline_rate_gap_mean={readout['decline_rate_gap_mean']:.6g} "
        f"decline_rate_gap_range={readout['decline_rate_gap_range']:.6g}",
        flush=True,
    )
    print(f"  non_degenerate={non_degenerate}  outcome={outcome}", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _t_start = time.perf_counter()
    manifest = run(dry_run=args.dry_run)

    out_path = write_flat_manifest(
        manifest,
        None,
        dry_run=args.dry_run,
        config=manifest.get("config"),
        seeds=manifest.get("seed_draw", {}).get("seeds_drawn") or [0],
        script_path=Path(__file__),
        started_at=_t_start,
    )
    print(f"\nResult written to: {out_path}", flush=True)

    _outcome = str(manifest.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
