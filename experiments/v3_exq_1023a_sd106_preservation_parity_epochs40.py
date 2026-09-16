"""V3-EXQ-1023a -- SD-106 ACCEPTANCE RE-MEASUREMENT AT P0a `epochs=40`: does generic bottleneck
variance preservation reach PCA-32 parity at the consumer rung once its P0a head is given the
training budget V3-EXQ-1041 showed it had not yet used?

Alphabetic-suffix iteration of V3-EXQ-1023 -- SAME QUESTION, SAME PRE-SET ACCEPTANCE TARGET,
SAME INSTRUMENT, RAISED BUDGET. Routed by the CONFIRMED autopsy
`failure_autopsy_V3-EXQ-1041_2026-09-16` (`recommended_writes_for_governance[10]`, user gate
2026-09-16T13:06:22Z, applied by governance-20260916, REE_assembly db6d20ebee).

  MANIPULATION: the SD-106 P0a head's TRAINING BUDGET -- `ZWorldP0Config.epochs` 12 -> 40, and
                nothing else. One integer. No new instrument, no new threshold, no new arm.
  HELD FIXED:   everything V3-EXQ-1023 held fixed, plus V3-EXQ-1023's own additions --
                `preservation_weight=200.0` (the V3-EXQ-1023 refusal to bump it STANDS),
                `use_world_encoder_skip=True`, the four tracks, the capacity ladder, the
                dataset recipe, the standardiser, the fit protocol, the calibration anchor,
                the acceptance bar, the seed-majority rule and the seeds [42, 43, 44].

EXPERIMENT_PURPOSE = "diagnostic". Stated rather than assumed, because this run DOES re-measure
a pre-set acceptance target: re-measuring a SUBSTRATE acceptance target is a READINESS verdict on
a landed design decision (/implement-substrate Step 8), not governance evidence for a mechanism
hypothesis. SD-106 is a design decision; a PASS here routes a BUILD decision, it does not weight
a claim's confidence. V3-EXQ-1023 was `diagnostic` for the same reason, and comparability with
the predecessor this run supersedes requires the same class.

SLEEP DRIVER: not applicable -- no sleep flag is set (the x734 all-ON stack at this rung enables
no sleep loop). Recorded as sleep_driver_pattern="none".

red-team: see the RED-TEAM RECORD at the end of this docstring and the queue entry note.

=== WHY epochs=40, AND WHY IT IS NOT A POWER-BUMP OF A BRAKED DESIGN ===

V3-EXQ-1041 (diagnostic PASS, confirmed) swept the P0a budget at IDENTICAL data and found:

  * the SD-106 preservation head has NOT saturated at the top of its sweep -- `sgd_head_r2`
    medians 0.746 (b_shipped, epochs 12) -> 0.806 (b_steps600, epochs 20) -> 0.854
    (b_steps1200, epochs 40), still rising at ~1160-1320 optimiser steps;
  * the mechanism's measured effect over its OWN paired OFF control TRIPLES across that sweep
    (+0.0169 -> +0.0315 -> +0.0509 median, positive on 3/3 seeds at every budget) while OFF
    parity stays flat (0.9245 / 0.9260 / 0.9208);
  * therefore V3-EXQ-1023's acceptance measurement measured this mechanism at roughly ONE THIRD
    of its within-range effect size.

So the shortfall V3-EXQ-1023 recorded is not yet attributable. This run takes the mechanism to
the top of the budget range V3-EXQ-1041 already executed six times (`b_steps1200` = 60 P0a
episodes x 40 epochs, exactly what `P0A_EPOCHS` reproduces) and asks the acceptance question
there. It is NOT a power-bump of a braked design: the re-derive brake on SD-106 stands at 0
(no `substrate_ceiling` autopsy on this claim), and V3-EXQ-1041's own routing names this run.

=== WHAT THIS RUN DISCRIMINATES, WITH PRE-REGISTERED QUANTITIES ===

Two readings of V3-EXQ-1023's consumer-rung shortfall survive V3-EXQ-1041. Both are legs on the
EXISTING question `zworld_actor_adequacy_locus`; this run opens no new leg and edits no registry.

  H-transfer-amplification  the preservation code's extra world-obs content TRANSFERS
                            monotonically to oracle-action agreement at the consumer rung.
                            PREDICTED consumer-rung agreement ~0.806.
  H-which-directions        the code preserves more VARIANCE but not more DECISION-RELEVANT
                            direction, so agreement barely moves.
                            PREDICTED consumer-rung agreement ~0.719.

`PREDICTED_TRANSFER_AMPLIFICATION` / `PREDICTED_WHICH_DIRECTIONS` below are those two numbers,
pre-registered, ~0.09 apart on the SHIPPED DV. They are RECORDED, NOT ROUTED ON: the verdict is
decided by the pre-set acceptance bar alone (C1), exactly as in V3-EXQ-1023. Recording them is
what lets /failure-autopsy read the outcome as a discrimination as well as an acceptance
verdict, without this run inventing a second threshold governance never set.

Note honestly: BOTH predictions sit BELOW `SD106_PARITY_BAR = 0.85`, so the most likely outcome
is a FAIL that nonetheless carries a discriminating reading. That is the designed value of the
run and is stated here so a FAIL is not read as a wasted cycle.

=== THE ACCEPTANCE TARGET, PRE-SET AND NOT INVENTED HERE ===

Transcribed through V3-EXQ-1023 from claims.yaml SD-106 `validation_experiment` and the
substrate_queue SD-106 entry, unchanged:

    the observation->z_world latent should reach PCA-32 parity at the consumer rung -- >= 0.85
    held-out oracle-action agreement at x734.PPOPolicyNet's PPO_TRUNK_HIDDEN on a seed majority

`SD106_PARITY_BAR`, `CONSUMER_RUNG` and `SEED_MAJORITY` are IMPORTED from x1023, never restated,
so this run cannot drift from the bar it is measured against.

=== WHY A THIN DRIVER AND NOT A COPY OF x1023 ===

The routing says "one config knob, no new driver". This file is the smallest honest expression
of that: it imports x1023 and replaces exactly TWO module-level functions --

  `_warm_sd106_agent`  identical to x1023's, except the SD-106 P0a config carries
                       `epochs=P0A_EPOCHS`;
  `_config_slice`      identical to x1023's, plus a `zworld_p0_epochs` key.

-- then calls `x1023.run_experiment` unchanged. Every threshold, gate, criterion, anchor
certificate and readout is x1023's own code, so a drift between this run and the acceptance
measurement it supersedes is structurally impossible rather than merely intended. x1023 is NOT
modified and continues to reproduce its own published numbers.

`zworld_p0_epochs` is added to the fingerprint slice for EVERY arm, not only the SD-106 track.
That is deliberate and conservative: a cell whose P0a budget differs is not the same
computation, and an omitted key is a false-HIT risk (a future consumer matching a 40-epoch cell
against a 12-epoch mint). Adding it for the OFF/anchor/raw arms too costs a false MISS -- free
here, since this lineage is Phase-0 emit-only and consumes no cached cell.

=== THE ONE-KNOB PROOF, AND WHY THE SMOKE CANNOT CARRY IT ===

`resolve_p0a_config(..., dry_run=True, ...)` FORCES `batch_size=8, epochs=2` (an intentional
shrink, `_lib/zworld_p0_warmup.py`), so a `--dry-run` smoke of this driver is byte-equivalent to
a smoke of x1023 and cannot witness `epochs=40`. The knob is proved in two other places instead:

  (1) `--self-test` asserts `resolve_p0a_config(seed=42, dry_run=False, config=P0A_CONFIG)`
      resolves to `epochs == 40` (and that x1023's own default resolves to 12), i.e. the knob
      reaches the trainer's own config object on the real path;
  (2) on a REAL run, the precondition `sd106_p0a_n_steps_supra_shipped` reads `p0a_n_steps`
      BACK OFF THE TRAINER for every SD-106 cell and gates on it. Below floor self-routes
      `substrate_not_ready_requeue` -- the knob did not engage is an INSTRUMENT failure, never a
      substrate verdict. That precondition is SCOPED OUT under `--dry-run` (disposition (a):
      not meaningful for a regime whose epochs are force-shrunk), never failed by it.

`P0A_STEP_FLOOR = 1000` is pre-registered from V3-EXQ-1041's own measured step counts
(b_shipped ~348-396 steps; b_steps1200 ~1160-1320 steps) and therefore sits strictly above the
shipped budget's ceiling and strictly below the raised budget's floor -- satisfiable by design-
time arithmetic, not by hope.

=== GFLAG-0286 DISCLOSURE: diverged rows are REPORTED, NOT EXCLUDED ===

`x1010._best_over_rungs` (ree-v3 experiments/v3_exq_1010_...py, the max-over-rungs helper) does
NOT filter rows whose decoder fit `diverged` (final CE at or above the uniform-logit value). This
run REPORTS them rather than excluding them, and the choice is deliberate in both directions:

  * EXCLUDING would silently change the instrument the acceptance target names as unchanged, and
    would make this run non-comparable with V3-EXQ-1023 and V3-EXQ-1010;
  * the hazard does not touch the verdict anyway. C1 -- the ONLY load-bearing criterion -- reads
    `x1010._cell(rows, <track>__mlp128, seed)`, the CONSUMER RUNG cell directly. It never calls
    `_best_over_rungs`. The best-over-rungs figures are recorded as context only.

So the manifest carries `diverged_cells` (count, and the offending arm/seed ids) and
`readout.n_diverged_cells`, and this docstring is the record of which option was taken.

=== V3-EXQ-1041 CONTEXT CARRIED FORWARD (recorded, not re-measured) ===

`prior_evidence_context` in the manifest carries V3-EXQ-1041's measured numbers for the very
cell this run's warmup reproduces, so a reader of THIS manifest does not have to re-derive them:

  * the in-training `preservation_holdout.r2` is a jointly-SGD-trained head's own holdout metric
    and UNDER-READS the same code on the same buffer and split by 0.0979 / 0.0967 / 0.1289
    absolute R^2 (seeds 42/43/44). It is a LOWER BOUND on linear-decodable content. This run
    records it per seed (`sd106_preservation_holdout_r2`) and it is NOT this run's DV.
  * under a post-hoc OLS probe the SD-106 code reaches 0.951 (as shipped) to 0.980 (at ~3.3x the
    optimiser steps -- i.e. at THIS run's budget) of the in-run PCA-32 achievable ceiling. That
    probe is NOT recomputed here: building it needs V3-EXQ-1041's observation-capture apparatus,
    which is a second instrument, not a config knob. Its values are cited from the run that
    measured exactly this cell six times.
  * the SD-106 design doc's headline 0.9974 / 0.9978 "obs R^2" table was measured on a DIFFERENT
    observation distribution: PCA-32 of the P0a rollout buffer reaches only 0.888 held-out /
    0.893 in-sample on seed 42 (cross-seed held-out median 0.886). Those absolutes are NOT
    comparable to any P0a acceptance number. Every criterion here is ceiling-relative or
    bar-relative, so the routing does not depend on it.
  * V3-EXQ-1041's shortfall decomposition is ORDER-DEPENDENT (metric-first routes
    `metrics_never_comparable`; budget-first routes `under_budgeted_p0a` at budget 0.769 median;
    order-symmetric routes its own tie branch `no_dominant_explanation`, shares metric 0.404 /
    budget 0.475 / residual 0.104). Recorded in `prior_evidence_context` so no reader of this
    manifest treats either partition as settled.

=== PRE-REGISTERED LIMITATION: THE OFF ARM IS NOT BUDGET-MATCHED (red-team F1) ===

STATED HERE SO NO READER OF THE MANIFEST HAS TO DISCOVER IT. The raised budget is applied to the
SD-106 track ONLY. `x1010._warm_off_agent` passes NO `zworld_p0_config`
(v3_exq_1010_...py:1087), so `resolve_p0a_config` takes its legacy branch
(_lib/zworld_p0_warmup.py) and the OFF arm trains at `ZWorldP0Config`'s default `epochs=12` --
the SHIPPED budget. Consequence, recorded rather than papered over:

  * the OFF track is a SHIPPED-BUDGET REFERENCE, reproducing V3-EXQ-1023's own OFF numbers at
    the same seeds. It is NOT a budget-matched control.
  * therefore every ON-minus-OFF quantity in this manifest -- `sd106_minus_off_consumer`,
    `arms_differ_detail.consumer_deltas` -- confounds {preservation term} with {3.3x more Adam
    steps on the shared SD-070 recipe}. Those deltas are CONTEXT ONLY and nothing routes on
    them. `off_arm_budget` in the manifest says so in machine-readable form, and `--self-test`
    asserts the asymmetry rather than leaving it to be inferred.

WHY THIS IS NOT REPAIRED BY ADDING AN OFF@40 ARM. Three reasons, in order of weight:
  (1) NOTHING LOAD-BEARING DEPENDS ON THE DELTA. C1, the only verdict-deciding criterion, is an
      ABSOLUTE bar (>= 0.85) against a PCA-32 anchor that is itself budget-independent (the
      anchor is a PCA of world_state, not a trained encoder). The two pre-registered
      discrimination predictions are likewise stated as consumer-rung agreement LEVELS
      (~0.806 / ~0.719), not as ON-minus-OFF deltas. So neither the acceptance verdict nor the
      discrimination is confounded.
  (2) A FOURTH WARMUP TRACK IS A NEW ARM, not a config knob. The routing this run implements
      (failure_autopsy_V3-EXQ-1041_2026-09-16, user-gated) specifies "one config knob, no new
      driver, the same 3 seeds, the exact b_steps1200 cell this run executed 6 times". Adding an
      OFF@40 track would expand a design governance already put in front of the user, and cost
      ~33% more warmup compute, to de-confound a quantity nothing routes on.
  (3) V3-EXQ-1041 ALREADY RAN THE MATCHED PAIR at this budget on ITS OWN DV: `off__b_steps1200`
      at n_steps 1320/1320/1160 beside `sd106__b_steps1200`, with OFF parity flat across the
      whole sweep (0.9245 / 0.9260 / 0.9208 / 0.9303). That is the R^2 DV, NOT the consumer-rung
      agreement DV, so it does not close the gap -- but it is why the gap is a recording
      limitation here rather than an open scientific question.

If a later reader wants the budget-matched consumer-rung contrast, that is a successor letter
with an OFF@40 track, and this block is the record of why it was not folded in here.

=== SEEDS ===

[42, 43, 44], IMPORTED from x1023. The standing caution about seed 44 on a reef-config env
(EXQ-539/540, V3-EXQ-538a) is acknowledged and deliberately NOT applied: V3-EXQ-1010,
V3-EXQ-1023 and V3-EXQ-1041 all completed on 42/43/44 at this rung and published per-seed values
for 44, and substituting 45 would break the seed-level pairing with the predecessor this run
supersedes.

=== RED-TEAM RECORD (Step 4.5) ===

See the queue entry note for the verdict, the reviewing model, and the disposition of every
finding.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.zworld_p0_warmup import resolve_p0a_config  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402

import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1010_zworld_overcapacity_decoder_sweep as x1010  # noqa: E402
import experiments.v3_exq_1023_sd106_bottleneck_preservation_validation as x1023  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1023a_sd106_preservation_parity_epochs40"
QUEUE_ID = "V3-EXQ-1023a"
SUPERSEDES = "V3-EXQ-1023"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-106"]

# ---- THE ONE KNOB --------------------------------------------------------------------
# V3-EXQ-1041's `b_steps1200` cell: 60 P0a episodes (unchanged) x 40 epochs. The SHIPPED value
# is ZWorldP0Config's own default, 12, which is what V3-EXQ-1023 ran at.
P0A_EPOCHS = 40
SHIPPED_P0A_EPOCHS = 12
# Pre-registered from V3-EXQ-1041's MEASURED step counts: b_shipped ~348-396, b_steps1200
# ~1160-1320 (`n_steps = (n_train // batch_size) * epochs`). Strictly above the shipped
# ceiling and strictly below the raised floor, so it is satisfiable by arithmetic.
P0A_STEP_FLOOR = 1000
# V3-EXQ-1041's OWN RECORDED per-seed `n_steps`, read from its manifest
# (v3_exq_1041_..._20260915T203743Z_v3.json, arm_results sd106__b_steps1200 / sd106__b_shipped,
# seeds 42/43/44). These are the known-positive and known-negative controls the step floor is
# certified against at setup by `assert_anchor_reachable` -- frozen literals, not re-derived.
REF_P0A_N_STEPS_EPOCHS40_1041 = [1320, 1320, 1160]     # the budget this run configures
REF_P0A_N_STEPS_SHIPPED_1041 = [396, 396, 348]         # the budget V3-EXQ-1023 ran at

# The AST scan cannot see through `x1023.run_cell`, which is where every cell of this run is
# actually executed and stamped: it enters `experiments._lib.arm_fingerprint.arm_cell(...)`
# per cell (complete RNG reset on enter + `cell.stamp(row)` on exit) with a config_slice this
# module supplies via `_config_slice_with_epochs`. So every `arm_results` row DOES carry an
# `arm_fingerprint`, and this run's slice differs from V3-EXQ-1023's by the added
# `zworld_p0_epochs` key -- the whole point of adding it.
ARM_FINGERPRINT_EXEMPT = (
    "cells are executed and stamped by x1023.run_cell's own arm_cell(...) context manager; "
    "this thin driver supplies the config_slice (with zworld_p0_epochs) and never opens a "
    "cell itself. Verified in the dry-run smoke: every arm_results row carries "
    "arm_fingerprint.substrate_hash.")

# ---- PRE-REGISTERED DISCRIMINATION PREDICTIONS (recorded, NOT routed on) ---------------
# PROVENANCE, stated because a bare literal is not a pre-registration:
#   0.806 -- failure_autopsy_V3-EXQ-1041_2026-09-16, recommended_substrate_queue_entry: "~0.806
#            under a monotone transfer". Numerically it is also V3-EXQ-1041's own
#            `sgd_head_r2` median at epochs=20, a DIFFERENT quantity on a different DV; the
#            autopsy's number is the authority here, not that coincidence.
#   0.719 -- the same entry's "~0.719 under a which-directions deficit". It is V3-EXQ-1023's
#            OWN observed SD-106 consumer-rung mean at epochs=12 (0.7192), i.e. literally the
#            "the raised budget changed nothing at the consumer rung" reading.
PREDICTED_TRANSFER_AMPLIFICATION = 0.806
PREDICTED_WHICH_DIRECTIONS = 0.719

# ---- IMPORTED, NEVER REDEFINED --------------------------------------------------------
SEEDS = x1023.SEEDS
DRY_RUN_SEEDS = x1023.DRY_RUN_SEEDS
SD106_PARITY_BAR = x1023.SD106_PARITY_BAR
PRESERVATION_WEIGHT = x1023.PRESERVATION_WEIGHT
USE_WORLD_ENCODER_SKIP = x1023.USE_WORLD_ENCODER_SKIP
CONSUMER_RUNG = x1023.CONSUMER_RUNG
SEED_MAJORITY = x1023.SEED_MAJORITY
TRACK_SD106 = x1023.TRACK_SD106
RUNG = x1023.RUNG
RUNG_ID = x1023.RUNG_ID

# V3-EXQ-1041's measured context for the cell this warmup reproduces. Recorded, not re-measured.
PRIOR_EVIDENCE_CONTEXT = {
    "source_run": "v3_exq_1041_sd106_preservation_step_budget_metric_diagnostic_"
                  "20260915T203743Z_v3",
    "source_autopsy": "failure_autopsy_V3-EXQ-1041_2026-09-16 (confirmed)",
    "budget_cell_reproduced_here": "b_steps1200 (60 P0a episodes x 40 epochs)",
    "sgd_head_r2_median_by_budget": {"epochs_12": 0.746, "epochs_20": 0.806, "epochs_40": 0.854},
    "sd106_minus_off_posthoc_median_by_budget": {"epochs_12": 0.0169, "epochs_20": 0.0315,
                                                 "epochs_40": 0.0509, "designref_100eps": 0.0277},
    "in_training_holdout_r2_underread_abs": {"42": 0.0979, "43": 0.0967, "44": 0.1289},
    "in_training_holdout_r2_note": (
        "sd106_preservation_holdout_r2 recorded by this run is a jointly-SGD-trained head's own "
        "holdout metric and is a LOWER BOUND on linear-decodable content; it is NOT this run's "
        "DV. The post-hoc OLS probe that reads the same code higher is V3-EXQ-1041's separate "
        "instrument and is cited, not recomputed here."),
    "posthoc_ols_fraction_of_pca32_ceiling": {"as_shipped": 0.951, "at_epochs_40": 0.980},
    "pca32_of_p0a_buffer_world_obs_r2_seed42": {"heldout": 0.888, "in_sample": 0.893,
                                                "cross_seed_heldout_median": 0.886},
    "design_doc_anchor_not_comparable": (
        "the SD-106 design doc's 0.9974/0.9978 obs R^2 table is on a DIFFERENT observation "
        "distribution; it is not comparable to any P0a acceptance number."),
    "decomposition_is_order_dependent": {
        "metric_first": "metrics_never_comparable",
        "budget_first": "under_budgeted_p0a (budget share 0.769 median)",
        "order_symmetric": ("no_dominant_explanation -- metric 0.404 / budget 0.475 / "
                            "residual 0.104, leader margin below the pre-registered 0.15 on "
                            "3/3 seeds"),
    },
}

# The ORIGINAL x1023 functions, captured at import time and BEFORE any override is installed.
# `_install_overrides` rebinds the module attributes, so an override that called
# `x1023._config_slice` by name would recurse into itself -- these two names are the only
# supported way to reach the originals.
_X1023_CONFIG_SLICE = x1023._config_slice
_X1023_WARM_SD106 = x1023._warm_sd106_agent

# The SD-106 P0a config THIS run trains with. Built once, at module scope, so the self-test and
# the warmup cannot drift apart.
P0A_CONFIG = ZWorldP0Config(preservation_weight=float(PRESERVATION_WEIGHT),
                            resource_field_weight=0.0,
                            epochs=int(P0A_EPOCHS))


# --------------------------------------------------------------------------------------
# THE TWO OVERRIDES
# --------------------------------------------------------------------------------------
def _warm_sd106_agent_epochs40(seed: int, env_kwargs: Dict[str, Any], sched: Dict[str, int],
                               dry_run: bool):
    """x1023._warm_sd106_agent, with `epochs=P0A_EPOCHS` on the SD-106 P0a config.

    Everything else -- the agent construction, the warmup family, the episode counts, the
    resource_field_weight, the encoder-health snapshot and the zero-init bypass check -- is
    x1023's, called through x1023's own helpers rather than re-implemented.
    """
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x1023._make_sd106_agent(warm_env)
    before = latent_stack_snapshot(agent)
    stats = x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=sched["p0"], p1_episodes=sched["p1"],
        steps_per_episode=sched["steps"], rung_id=RUNG_ID,
        total_denominator=(sched["p0"] + sched["p1"]),
        zworld_p0_episodes=sched["zworld_p0"],
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if sched["zworld_p0"] > 0 else None),
        zworld_p0_dry_run=dry_run,
        # THE ONE KNOB. resource_field_weight stays 0.0 and is carried INSIDE the config --
        # run_zworld_p0 refuses both sources at once.
        zworld_p0_config=P0A_CONFIG,
    )
    guard = latent_stack_weight_delta(agent, before)
    x1023._ZG.observe(agent)
    skip = getattr(agent.latent_stack.split_encoder, "world_encoder_skip", None)
    skip_norm = float(skip.weight.detach().norm()) if skip is not None else 0.0
    return agent, stats, guard, skip_norm


def _config_slice_with_epochs(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    """x1023._config_slice plus `zworld_p0_epochs`, on EVERY arm (see the docstring)."""
    d = _X1023_CONFIG_SLICE(base, arm_id)
    ctx = x1023._ctx(arm_id)
    d["zworld_p0_epochs"] = (int(P0A_EPOCHS) if ctx["is_sd106_track"]
                             else int(SHIPPED_P0A_EPOCHS))
    d["zworld_p0_epochs_manipulated"] = True
    return d


def _install_overrides() -> None:
    """Idempotent. Rebinds the two x1023 module globals `run_cell` looks up."""
    x1023._warm_sd106_agent = _warm_sd106_agent_epochs40
    x1023._config_slice = _config_slice_with_epochs


# --------------------------------------------------------------------------------------
# POST-PROCESSING: the budget precondition, the GFLAG-0286 disclosure, the predictions
# --------------------------------------------------------------------------------------
def _step_floor_predicate(n_steps: Any) -> bool:
    """THE SHIPPED PREDICATE for the budget precondition. Used by `_postprocess` (through
    `_worst_p0a_steps`, whose min-over-cells IS this predicate applied to every cell) and by
    `_certify_step_floor` -- one definition, so the certificate cannot score a copy."""
    return n_steps is not None and float(n_steps) >= float(P0A_STEP_FLOOR)


def _certify_step_floor() -> Dict[str, Any]:
    """Refuse, at setup, a step floor V3-EXQ-1041's own epochs=40 cells could not clear.

    The precondition self-routes `substrate_not_ready_requeue`, so a floor NARROWER than the
    state it anchors to would report met=false on every run forever and mislabel an
    instrument-specification gap as a substrate verdict. `threshold=1.0` because the shipped
    predicate is a WORST-CELL (min-over-cells) claim: every cell must clear.
    """
    return assert_anchor_reachable(
        anchor_name="sd106_p0a_n_steps_supra_shipped",
        reference_cells=list(REF_P0A_N_STEPS_EPOCHS40_1041),
        score_fn=_step_floor_predicate,
        threshold=1.0,
        reference_source=("V3-EXQ-1041 manifest arm_results sd106__b_steps1200, seeds "
                          "42/43/44 -- the same 60-episode x 40-epoch cell this run's warmup "
                          "reproduces"),
    )


def _sd106_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("track") == TRACK_SD106]


def _worst_p0a_steps(rows: List[Dict[str, Any]]
                     ) -> Tuple[Optional[int], Optional[str], int, int]:
    """The WORST `p0a_n_steps` over the SD-106 cells, the offending cell id, and the counts.

    `met` is a worst-case claim, so the reported number is the extremum, not the mean -- an
    in-band mean would mask an out-of-band cell and the indexer would recompute MET against our
    own met=False.

    RED-TEAM F3. A cell that reported NO `p0a_n_steps` at all (the trainer refused the buffer:
    `run_zworld_p0` sets `p0a_ran=False` and never writes the key) must NOT simply drop out of
    the minimum -- two good seeds and one refused seed would otherwise record
    `measured=1320, passed=True`. A non-reporting cell IS the worst possible cell, so it scores
    as 0 steps. That keeps `measured` the SAME STATISTIC `met` tests, which is what stops the
    indexer recomputing MET against our own met=False.
    """
    worst, worst_cell, n_cells, n_missing = None, None, 0, 0
    for r in _sd106_rows(rows):
        n_cells += 1
        p0a = (r.get("warmup_stats") or {}).get("zworld_p0") or {}
        raw = p0a.get("p0a_n_steps")
        cell_id = "%s@seed%s" % (r.get("arm_id"), r.get("seed"))
        if raw is None:
            n_missing += 1
            n = 0                      # a cell that never reported is the worst cell
            cell_id += " (no p0a_n_steps reported)"
        else:
            n = int(raw)
        if worst is None or n < worst:
            worst, worst_cell = n, cell_id
    return worst, worst_cell, n_cells, n_missing


def _diverged_cells(rows: List[Dict[str, Any]]) -> List[str]:
    """Every cell whose decoder fit reported `diverged` true, anywhere in its recorded stats."""
    out: List[str] = []

    def _has_diverged(o: Any) -> bool:
        if isinstance(o, dict):
            if o.get("diverged") is True:
                return True
            return any(_has_diverged(v) for v in o.values())
        if isinstance(o, list):
            return any(_has_diverged(v) for v in o)
        return False

    for r in rows:
        if _has_diverged(r):
            out.append("%s@seed%s" % (r.get("arm_id"), r.get("seed")))
    return sorted(out)


def _postprocess(result: Dict[str, Any], dry_run: bool,
                 step_floor_certificate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append this iteration's own records to x1023's result. Never rewrites its criteria."""
    rows = result.get("arm_results") or []
    interp = result.setdefault("interpretation", {})
    preconds = interp.setdefault("preconditions", [])
    criteria = interp.setdefault("criteria", [])
    nondeg = interp.setdefault("criteria_non_degenerate", {})
    flat = result.setdefault("readout", {})

    # ---- (1) the budget precondition -- REAL RUNS ONLY (disposition (a): scoped out, not
    #          failed, under a regime whose epochs are force-shrunk by the dry-run path).
    worst_steps, worst_cell, n_sd106_cells, n_missing = _worst_p0a_steps(rows)
    # RED-TEAM F2. "No SD-106 cell ran at all" and "the knob did not engage" are DIFFERENT
    # facts with different repairs, and x1023's gate legitimately short-circuits every swept
    # cell when the instrument or anchor precondition goes red. Overwriting its gate_reason
    # with "the epochs=40 knob did not reach the trainer" in that case records a false cause
    # and would send a later autopsy to the wrong instrument. So the override below fires ONLY
    # when SD-106 cells actually ran.
    sd106_cells_ran = bool(n_sd106_cells > 0)
    budget_engaged: Optional[bool] = None
    if dry_run:
        interp.setdefault("scoped_out", []).append({
            "name": "sd106_p0a_n_steps_supra_shipped",
            "applies_note": ("scoped out under --dry-run: resolve_p0a_config FORCES epochs=2 "
                             "and batch_size=8 on the dry path, so the raised budget is not "
                             "meaningful for this regime. The knob is proved on the dry path "
                             "by --self-test instead."),
        })
    elif not sd106_cells_ran:
        interp.setdefault("scoped_out", []).append({
            "name": "sd106_p0a_n_steps_supra_shipped",
            "applies_note": ("scoped out: NO SD-106 cell ran, so there is no budget to read "
                             "back. x1023's own gate short-circuited the swept arms -- its "
                             "gate_reason, not this precondition, carries the real cause."),
        })
    else:
        budget_engaged = bool(_step_floor_predicate(worst_steps) and n_missing == 0)
        preconds.append({
            "name": "sd106_p0a_n_steps_supra_shipped",
            "description": ("the SD-106 P0a head actually took the raised optimiser-step "
                            "budget -- read BACK OFF THE TRAINER as p0a_n_steps, not asserted"),
            "measured": (int(worst_steps) if worst_steps is not None else None),
            "threshold": int(P0A_STEP_FLOOR),
            "direction": "lower",
            "offending_cell": worst_cell,
            "n_sd106_cells": int(n_sd106_cells),
            "n_cells_missing_p0a_n_steps": int(n_missing),
            "control": ("worst (minimum) p0a_n_steps over every SD-106 cell, with a "
                        "non-reporting cell scored as 0. V3-EXQ-1041 measured 396/396/348 "
                        "steps at the shipped epochs=12 and 1320/1320/1160 at epochs=40, so "
                        "this floor separates the two budgets by construction."),
            "met": bool(budget_engaged),
        })
        if not budget_engaged:
            # The knob did not engage. That is an INSTRUMENT failure, never a substrate verdict.
            interp["label"] = "substrate_not_ready_requeue"
            result["outcome"] = "FAIL"
            result["non_degenerate"] = False
            result["degeneracy_reason"] = (
                "SD-106 P0a budget did not engage: worst p0a_n_steps %r < floor %d (%s); "
                "%d of %d SD-106 cells reported no step count. The epochs=40 knob did not "
                "reach the trainer, so nothing about SD-106's parity was measured."
                % (worst_steps, int(P0A_STEP_FLOOR), worst_cell, n_missing, n_sd106_cells))

    # C3 mirrors the precondition. When the precondition is scoped out there is no bar to score
    # against, so `threshold_not_applicable` is declared rather than a numeric pair the indexer
    # would recompute MET from and disagree with (red-team note on the dry-run manifest).
    if budget_engaged is None:
        criteria.append({
            "name": "C3_p0a_budget_raised_above_shipped",
            "load_bearing": False,
            "passed": False,
            "threshold_not_applicable": (
                "scoped out: the raised budget was not in force for this run "
                "(dry_run=%s, sd106_cells_ran=%s). Nothing was measured, so no bar applies."
                % (bool(dry_run), sd106_cells_ran)),
            "detail": {"epochs_configured": int(P0A_EPOCHS),
                       "epochs_shipped_in_V3-EXQ-1023": int(SHIPPED_P0A_EPOCHS),
                       "n_sd106_cells": int(n_sd106_cells)},
        })
    else:
        criteria.append({
            "name": "C3_p0a_budget_raised_above_shipped",
            "load_bearing": False,
            "passed": bool(budget_engaged),
            "measured": (int(worst_steps) if worst_steps is not None else None),
            "threshold": int(P0A_STEP_FLOOR),
            "offending_cell": worst_cell,
            "detail": {"epochs_configured": int(P0A_EPOCHS),
                       "epochs_shipped_in_V3-EXQ-1023": int(SHIPPED_P0A_EPOCHS),
                       "n_sd106_cells": int(n_sd106_cells),
                       "n_cells_missing_p0a_n_steps": int(n_missing)},
            "threshold_note": ("non-degeneracy witness for the MANIPULATION: the raised budget "
                               "must have reached the trainer on EVERY SD-106 cell. Does not "
                               "decide the verdict -- C1 does."),
        })
    nondeg["C3_p0a_budget_raised_above_shipped"] = bool(budget_engaged is not None)

    # ---- (1b) RED-TEAM F1: the OFF arm's budget, recorded in machine-readable form ---------
    result["off_arm_budget"] = {
        "sd106_track_p0a_epochs": int(P0A_EPOCHS),
        "off_track_p0a_epochs": int(SHIPPED_P0A_EPOCHS),
        "budget_matched": False,
        "why": ("x1010._warm_off_agent passes no zworld_p0_config, so the OFF track trains at "
                "ZWorldP0Config's default epochs=12 -- the SHIPPED budget. The OFF track is a "
                "shipped-budget REFERENCE reproducing V3-EXQ-1023's own OFF numbers at the "
                "same seeds, NOT a budget-matched control."),
        "consequence": ("every ON-minus-OFF quantity in this manifest confounds the "
                        "preservation term with 3.3x more Adam steps. Those deltas are CONTEXT "
                        "ONLY. Nothing routes on them: C1 is an absolute bar against a "
                        "budget-independent PCA-32 anchor, and both pre-registered "
                        "discrimination predictions are agreement LEVELS, not deltas."),
        "successor_note": ("a budget-matched consumer-rung contrast needs an OFF@40 track, "
                           "which is a new arm and therefore a successor letter, not this "
                           "run's one-config-knob routing."),
    }
    flat["off_track_p0a_epochs"] = int(SHIPPED_P0A_EPOCHS)
    flat["on_off_budget_matched"] = 0

    # ---- (2) GFLAG-0286: diverged rows REPORTED, not excluded (see the docstring) --------
    dv = _diverged_cells(rows)
    result["diverged_cells"] = {
        "n": len(dv),
        "cells": dv,
        "disposition": "reported_not_excluded",
        "why": ("x1010._best_over_rungs applies no diverged filter. EXCLUDING would change the "
                "instrument the acceptance target names as unchanged and break comparability "
                "with V3-EXQ-1023/1010. The load-bearing criterion C1 reads the CONSUMER RUNG "
                "cell directly (x1010._cell), never _best_over_rungs, so the hazard cannot "
                "touch the verdict; best-over-rungs figures are context only."),
    }
    flat["n_diverged_cells"] = int(len(dv))

    # ---- (3) the pre-registered discrimination predictions, recorded not routed on -------
    obs = flat.get("sd106_consumer_agreement_mean")
    result["discrimination"] = {
        "question_id": "zworld_actor_adequacy_locus",
        "legs_adjudicated": ["H-transfer-amplification", "H-which-directions"],
        "predicted": {"H-transfer-amplification": float(PREDICTED_TRANSFER_AMPLIFICATION),
                      "H-which-directions": float(PREDICTED_WHICH_DIRECTIONS)},
        "predicted_separation": round(float(PREDICTED_TRANSFER_AMPLIFICATION)
                                      - float(PREDICTED_WHICH_DIRECTIONS), 4),
        "observed_sd106_consumer_agreement_mean": obs,
        "routed_on": ("NOTHING. The verdict is decided by C1 against the pre-set acceptance "
                      "bar alone. These predictions are recorded so /failure-autopsy can read "
                      "the outcome as a discrimination without this run inventing a second "
                      "threshold governance never set. Both predictions sit BELOW the 0.85 "
                      "bar, so a FAIL here is an informative discrimination, not a null."),
        "registry_note": ("both legs are EXISTING legs on the EXISTING question; this run opens "
                          "no leg and edits no registry file."),
    }
    flat["predicted_transfer_amplification"] = float(PREDICTED_TRANSFER_AMPLIFICATION)
    flat["predicted_which_directions"] = float(PREDICTED_WHICH_DIRECTIONS)
    flat["p0a_epochs"] = int(P0A_EPOCHS)
    flat["p0a_epochs_shipped_predecessor"] = int(SHIPPED_P0A_EPOCHS)
    if worst_steps is not None and sd106_cells_ran:
        flat["p0a_n_steps_worst_sd106_cell"] = int(worst_steps)
        flat["n_sd106_cells_missing_p0a_n_steps"] = int(n_missing)

    # ---- (3b) the setup-time reachability certificate for this run's own anchor ----------
    if step_floor_certificate is not None:
        anchors = interp.get("anchor_reachability")
        if isinstance(anchors, list):
            anchors.append(step_floor_certificate)
        elif isinstance(anchors, dict):
            interp["anchor_reachability"] = [anchors, step_floor_certificate]
        else:
            interp["anchor_reachability"] = [step_floor_certificate]

    # ---- (4) provenance + carried context ------------------------------------------------
    result["prior_evidence_context"] = PRIOR_EVIDENCE_CONTEXT
    result["supersedes"] = SUPERSEDES
    result["experiment_type"] = EXPERIMENT_TYPE
    result["experiment_purpose"] = EXPERIMENT_PURPOSE
    result["claim_ids"] = list(CLAIM_IDS)

    # Drop non-finite / None from the flat block (a nan is numeric to the indexer).
    result["readout"] = {k: v for k, v in flat.items()
                         if v is not None and (not isinstance(v, float) or v == v)}
    return result


# --------------------------------------------------------------------------------------
def _run_self_test() -> int:
    fails = 0

    def _chk(ok: bool, msg: str) -> None:
        nonlocal fails
        if not ok:
            fails += 1
            print("[self-test] FAIL: %s" % msg, flush=True)

    # (1) THE ONE-KNOB PROOF -- the knob reaches the trainer's own resolved config.
    real = resolve_p0a_config(seed=42, dry_run=False, resource_field_weight=0.0,
                              config=P0A_CONFIG)
    _chk(int(real.epochs) == int(P0A_EPOCHS),
         "resolved non-dry epochs is %r, expected %d" % (real.epochs, P0A_EPOCHS))
    _chk(float(real.preservation_weight) == float(PRESERVATION_WEIGHT),
         "resolved preservation_weight drifted: %r" % (real.preservation_weight,))
    _chk(float(real.resource_field_weight) == 0.0,
         "resolved resource_field_weight must stay 0.0 (978's OFF arm)")
    _chk(int(real.seed) == 42, "resolve_p0a_config must stamp the warmup seed")
    shipped = resolve_p0a_config(seed=42, dry_run=False, resource_field_weight=0.0,
                                 config=ZWorldP0Config(
                                     preservation_weight=float(PRESERVATION_WEIGHT),
                                     resource_field_weight=0.0))
    _chk(int(shipped.epochs) == int(SHIPPED_P0A_EPOCHS),
         "the predecessor's default epochs is %r, expected %d -- the 12-vs-40 contrast this "
         "run rests on has drifted" % (shipped.epochs, SHIPPED_P0A_EPOCHS))

    # (2) the step floor is SATISFIABLE and DISCRIMINATING against V3-EXQ-1041's own cells.
    #     Reachability (positive control) is certified by the same helper the run calls at
    #     setup; the negative control below is what stops the floor being trivially true.
    cert = _certify_step_floor()
    _chk(bool(cert.get("reachable")),
         "the step floor is unreachable by V3-EXQ-1041's own epochs=40 cells: %r" % (cert,))
    _chk(not any(_step_floor_predicate(v) for v in REF_P0A_N_STEPS_SHIPPED_1041),
         "NEGATIVE CONTROL: the step floor must REJECT V3-EXQ-1023's shipped budget "
         "(%r) -- a floor both budgets clear witnesses nothing"
         % (REF_P0A_N_STEPS_SHIPPED_1041,))

    # (3) the overrides actually install, and the slice changes.
    _install_overrides()
    _chk(x1023._warm_sd106_agent is _warm_sd106_agent_epochs40,
         "the warmup override did not install on x1023")
    _chk(x1023._config_slice is _config_slice_with_epochs,
         "the config-slice override did not install on x1023")
    on = _config_slice_with_epochs({}, x1023._arm_id(TRACK_SD106, CONSUMER_RUNG))
    off = _config_slice_with_epochs({}, x1023._arm_id(x1023.TRACK_OFF, CONSUMER_RUNG))
    _chk(int(on.get("zworld_p0_epochs", -1)) == int(P0A_EPOCHS),
         "the SD-106 slice does not carry zworld_p0_epochs=%d" % P0A_EPOCHS)
    _chk(on != off, "the SD-106 and OFF fingerprint slices must not collide")
    _chk(float(on.get("preservation_weight", -1)) == float(PRESERVATION_WEIGHT),
         "the SD-106 slice lost preservation_weight -- x1023._config_slice drifted")

    # (4) the acceptance target is x1023's, unmodified.
    _chk(float(SD106_PARITY_BAR) == 0.85, "SD106_PARITY_BAR drifted from the pre-set 0.85")
    _chk(str(CONSUMER_RUNG) == "mlp128", "CONSUMER_RUNG drifted from mlp128")
    _chk(int(SEED_MAJORITY) == 2, "SEED_MAJORITY drifted from 2")
    _chk(float(PRESERVATION_WEIGHT) == 200.0,
         "PRESERVATION_WEIGHT drifted -- the V3-EXQ-1023 refusal to bump it stands")
    _chk(list(SEEDS) == [42, 43, 44], "SEEDS drifted from the paired [42, 43, 44]")

    # (5) RED-TEAM F1's own confirmer, shipped as an assertion: the OFF track's budget is
    #     ASYMMETRIC to the SD-106 track's, and that asymmetry is a recorded fact rather than
    #     something a later reader has to rediscover from x1010's call site.
    off_resolved = resolve_p0a_config(seed=42, dry_run=False, resource_field_weight=0.0)
    _chk(int(off_resolved.epochs) == int(SHIPPED_P0A_EPOCHS),
         "the OFF track (x1010._warm_off_agent, no zworld_p0_config) must resolve to the "
         "shipped epochs=%d; got %r. If this changed, the OFF arm is no longer the "
         "shipped-budget reference the docstring documents."
         % (SHIPPED_P0A_EPOCHS, off_resolved.epochs))
    _chk(int(off_resolved.epochs) != int(P0A_EPOCHS),
         "OFF and SD-106 budgets must differ -- this run's whole ON/OFF limitation block "
         "assumes they do")
    _chk(float(off_resolved.preservation_weight) == 0.0,
         "the OFF track must carry NO preservation term")

    # (6) RED-TEAM F2's confirmer: a red x1023 gate that ran NO SD-106 cell must NOT be
    #     relabelled 'the epochs=40 knob did not reach the trainer'.
    stub = {"arm_results": [],
            "readout": {},
            "outcome": "FAIL",
            "non_degenerate": False,
            "degeneracy_reason": "preconditions unmet",
            "interpretation": {"label": "substrate_not_ready_requeue",
                               "gate_reason": ("preconditions unmet: "
                                               "instrument_rawfield_control_supra_floor"),
                               "preconditions": [], "criteria": [],
                               "criteria_non_degenerate": {}}}
    out = _postprocess(stub, dry_run=False)
    _chk("knob did not reach the trainer" not in str(out.get("degeneracy_reason")),
         "F2 regression: a no-SD-106-cell gate red was relabelled as a knob failure")
    _chk(str(out["interpretation"].get("gate_reason", "")).startswith("preconditions unmet"),
         "F2 regression: x1023's own gate_reason was overwritten")
    c3 = [c for c in out["interpretation"]["criteria"]
          if c["name"] == "C3_p0a_budget_raised_above_shipped"]
    _chk(len(c3) == 1 and "threshold" not in c3[0],
         "F2/dry regression: a scoped-out C3 must declare threshold_not_applicable, never a "
         "numeric bar the indexer would recompute MET from")

    # (7) RED-TEAM F3's confirmer: a cell that reported no step count is the WORST cell, not a
    #     cell that silently drops out of the minimum.
    _rows = [{"track": TRACK_SD106, "arm_id": "a", "seed": 42,
              "warmup_stats": {"zworld_p0": {"p0a_n_steps": 1320}}},
             {"track": TRACK_SD106, "arm_id": "b", "seed": 43,
              "warmup_stats": {"zworld_p0": {"p0a_ran": False}}}]
    w, cellid, n_cells, n_missing = _worst_p0a_steps(_rows)
    _chk(w == 0 and n_missing == 1 and n_cells == 2 and "b@seed43" in str(cellid),
         "F3 regression: a non-reporting SD-106 cell dropped out of the worst-cell minimum "
         "(got worst=%r cell=%r n_cells=%r n_missing=%r)" % (w, cellid, n_cells, n_missing))
    _chk(not _step_floor_predicate(w),
         "F3 regression: the worst-cell value must FAIL the step floor when a cell is missing")

    # (8) x1023's own self-test still passes under our overrides (it asserts the ON/OFF slices
    #     differ and the gates are reachable) -- a drift in the borrowed instrument is ours too.
    _chk(x1023._run_self_test() == 0, "x1023's own self-test failed under these overrides")

    print("[self-test] %d failure(s)" % fails, flush=True)
    return fails


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    _install_overrides()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else list(SEEDS))
    print("%s: seeds=%s dry_run=%s" % (EXPERIMENT_TYPE, seeds, bool(args.dry_run)), flush=True)
    _resolved = resolve_p0a_config(seed=int(seeds[0]), dry_run=bool(args.dry_run),
                                   resource_field_weight=0.0, config=P0A_CONFIG)
    print("[knob] P0A_EPOCHS=%d shipped=%d -> resolved epochs=%d batch_size=%d (dry_run=%s)"
          % (P0A_EPOCHS, SHIPPED_P0A_EPOCHS, int(_resolved.epochs), int(_resolved.batch_size),
             bool(args.dry_run)), flush=True)

    # Setup-time refusal BEFORE any compute: a floor V3-EXQ-1041's own epochs=40 cells could
    # not clear would report met=false forever and mislabel an instrument gap as a verdict.
    _step_cert = _certify_step_floor()
    print("[anchor] sd106_p0a_n_steps_supra_shipped reachable=%s reference_score=%s"
          % (_step_cert.get("reachable"), _step_cert.get("reference_score")), flush=True)

    result = x1023.run_experiment(list(seeds), dry_run=bool(args.dry_run))
    result = _postprocess(result, dry_run=bool(args.dry_run),
                          step_floor_certificate=_step_cert)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    full_config = {
        "rung": RUNG, "level_id": x1023.LEVEL_ID,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "zworld_p0_episodes": (x1023.DRY_RUN_ZWORLD_P0 if args.dry_run
                               else x1023.ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (x1023.DRY_RUN_P0 if args.dry_run else x1023.P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (x1023.DRY_RUN_P1 if args.dry_run
                                  else x1023.P1_REINFORCE_EPISODES),
        "eval_episodes": (x1023.DRY_RUN_EVAL if args.dry_run else x1023.EVAL_EPISODES),
        "steps_per_episode": (x1023.DRY_RUN_STEPS if args.dry_run else x1023.STEPS_PER_EPISODE),
        "bc_episodes": (x1023.DRY_RUN_BC_EPISODES if args.dry_run else x1023.BC_EPISODES),
        "bc_random_episodes": (x1023.DRY_RUN_BC_RANDOM_EPISODES if args.dry_run
                               else x1023.BC_RANDOM_EPISODES),
        "bc_train_frac": x1002.BC_TRAIN_FRAC,
        "adapter_passes": (x1023.DRY_RUN_ADAPTER_PASSES if args.dry_run
                           else x1023.ADAPTER_PASSES),
        "adapter_batch": x1023.ADAPTER_BATCH, "adapter_lr": x1023.ADAPTER_LR,
        # THE MANIPULATED VARIABLE.
        "sd106_p0a_epochs": int(P0A_EPOCHS),
        "sd106_p0a_epochs_shipped_predecessor": int(SHIPPED_P0A_EPOCHS),
        "sd106_p0a_step_floor": int(P0A_STEP_FLOOR),
        # HELD FIXED, and declared so a later reader can see it was not moved.
        "sd106_preservation_weight": float(PRESERVATION_WEIGHT),
        "sd106_use_world_encoder_skip": bool(USE_WORLD_ENCODER_SKIP),
        "sd106_parity_bar": float(SD106_PARITY_BAR),
        "capacity_ladder": [{"rung": r, "kind": k, "hidden": h, "depth": d}
                            for r, k, h, d in x1023.CAPACITY_LADDER],
        "capacity_consumer_rung": CONSUMER_RUNG,
        "capacity_max_rung": x1023.MAX_CAPACITY_RUNG,
        "consumer_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "tracks": list(x1023.SWEPT_TRACKS),
        "projection_dim": x1023.PROJECTION_DIM,
        "raw_field_control_floor": x1023.RAW_FIELD_CONTROL_FLOOR,
        "seed_majority": int(SEED_MAJORITY), "arms": x1023.ARM_IDS,
        "supersedes": SUPERSEDES,
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=x1023._ZG.stats(),
    )
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]),
          flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
