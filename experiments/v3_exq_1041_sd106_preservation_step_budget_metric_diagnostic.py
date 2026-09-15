"""V3-EXQ-1041 -- SD-106 DIAGNOSTIC: is the 0.70-0.75 preservation holdout R^2 a METRIC
MISMATCH, an UNDER-BUDGETED P0a, or a MECHANISM DEFECT?

This is the cheap, P0-warmup-only diagnostic the CONFIRMED
`failure_autopsy_V3-EXQ-1023_2026-09-14.json` (status `confirmed`, REE_assembly `beb47bca09`)
routes for SD-106 BEFORE any `preservation_weight` change and BEFORE re-running the acceptance
measurement. Its `recommended_substrate_queue_entry.implementation_hint` is the spec this
driver implements, transcribed below rather than paraphrased.

SLEEP DRIVER: not applicable -- no sleep flag is set anywhere in this driver. Recorded as
sleep_driver_pattern="none".

red-team (fable): see the RED-TEAM RECORD at the end of this docstring and the queue entry note.

=== THE OBSERVATION THIS EXISTS TO EXPLAIN ===

V3-EXQ-1023 (diagnostic FAIL, `..._20260912T045319Z_v3`) ran the SD-106 acceptance measurement
at `preservation_weight=200.0` + `use_world_encoder_skip=True` and recorded, per seed:

    seed 42: preservation_holdout.r2 = 0.7490045122982752   (n_steps 396, n_buffered 2716)
    seed 43: preservation_holdout.r2 = 0.7463981728369910   (n_steps 396, n_buffered 2671)
    seed 44: preservation_holdout.r2 = 0.6998874112632603   (n_steps 348, n_buffered 2364)

The SD-106 design doc (`REE_assembly/docs/architecture/sd_106_generic_bottleneck_variance_
preservation.md`, "Measured effect") records, for the SAME recipe, LINEAR-DECODABLE `world_obs`
R^2 of **0.9974 / 0.9978** against a PCA-32 anchor of 0.9983 / 0.9984 -- and 0.9432 / 0.9493 for
the SHIPPED (no-preservation) recipe. A ~0.25 gap between "the same quantity" measured two ways.

THE TWO READINGS ARE NOT THE SAME STATISTIC, and that is the first hypothesis:

  (a) `preservation_holdout.r2` (`ree_core/latent/zworld_p0.py`, the `stats["preservation_
      holdout"]` block) is computed from `self._preserve_head`, an `nn.Linear` trained BY SGD
      JOINTLY WITH THE ENCODER over `n_steps` Adam steps. Its R^2 is bounded by BOTH how much
      variance the code preserves AND how converged that head is.
  (b) The design-time figure is a POST-HOC linear probe: the best linear map from the frozen
      code to `world_obs`. It removes the head-convergence term entirely.

(b) >= (a) on the train split BY CONSTRUCTION (OLS is the MSE optimum of the same linear
family), so the GAP between them IS the head-convergence deficit, measured directly.

The budget is the second hypothesis. The design doc's own cost table names "a realistic P0 step
count (600 steps, 4000 buffered observations)". V3-EXQ-1023 ran at 348-396 steps on 2364-2716
observations -- roughly 60% of the doc's own reference on BOTH coordinates. The doc does NOT
record the budget behind the 0.9974/0.9978 table itself (the design-time probe was a throwaway
and is not in the repo -- checked at `ree-v3` HEAD and in the SD-106 landing commit 616e713),
so that reference point has to be RECONSTRUCTED, which this run does.

=== WHAT THIS RUN MEASURES ===

On the SAME frozen SD-106 latent at weight=200, per cell, BOTH readouts on the SAME held-out
split:

  (a) `sgd_head_r2`      -- `preservation_holdout.r2` straight out of the canonical P0a path.
  (b) `posthoc_ols_r2`   -- a post-hoc OLS linear probe z -> world_obs, fitted on the TRAIN
                            split and scored on the HELD-OUT split against the train-split mean
                            predictor. Byte-for-byte the same R^2 DEFINITION the substrate uses
                            for (a); only the fit differs (closed-form optimum vs SGD).
  (c) `pca32_ols_r2`     -- THE ACHIEVABLE CEILING. The identical OLS probe applied to PCA-32
                            of the same `world_obs` on the same split: the best any 32-dim
                            linear code can do on this data (Eckart-Young). Every criterion is
                            a RATIO to it, so no bar can be unreachable because of the dataset.
  (d) `identity_probe_r2`-- THE INSTRUMENT GATE. The same probe with `world_obs` itself as the
                            code. The identity map is inside the OLS family, so this is ~1.0 on
                            ANY data; it fails only if the probe's arithmetic, the split or the
                            numerics are broken. Data-independent on purpose -- an absolute
                            floor on (c) would gate the science on the dataset, not the probe.

NO decoder ladder, NO oracle-adapter dataset collection, NO P0b, NO P1 -- exactly as the
autopsy's hint requires. The entire run is `run_zworld_p0` plus arithmetic.

=== THE ARMS (2 tracks x 4 budgets x 3 seeds = 24 cells) ===

TRACKS (the SD-106 manipulation, imported not re-derived):
  `sd106`  `_make_sd106_agent` from V3-EXQ-1023 (x724 all-ON + `use_world_encoder_skip=True`),
           P0a at `ZWorldP0Config(preservation_weight=200.0)`. ** THE SUBJECT. **
  `off`    `x1002._make_agent` (978's OFF arm, the pre-SD-106 recipe), P0a at
           `preservation_weight=0.0`. ** THE PAIRED REFERENCE. ** It supplies the design doc's
           0.9432/0.9493 shipped row, so the run can say whether the preservation term RAISES
           or LOWERS post-hoc linear decodability -- which (a) alone cannot, because the OFF
           arm builds no `_preserve_head` and therefore reports no (a) at all.

BUDGETS (episodes x epochs; `n_steps = (n_train // batch_size) * epochs`):
  `b_shipped`    60 eps, epochs 12 -> ~348-396 steps, ~2400-2700 obs
                 ** V3-EXQ-1023'S EXACT CONFIG. ** Its job is the REPRODUCTION GATE below.
  `b_steps600`   60 eps, epochs 20 -> ~580-660 steps, SAME obs   (steps varied, data held)
  `b_steps1200`  60 eps, epochs 40 -> ~1160-1320 steps, SAME obs (steps varied, data held)
  `b_designref` 100 eps, epochs 12 -> ~600 steps, ~4000-4500 obs
                 ** THE DESIGN DOC'S LITERAL REFERENCE POINT ** (600 steps, 4000 observations).

The 2x2 that matters: `b_steps600` and `b_designref` sit at ~the same STEP count and differ ~2x
in DATA, while `b_shipped`/`b_steps600`/`b_steps1200` differ ~3.3x in STEPS at IDENTICAL data.
So a budget effect is attributable to steps or to data rather than to "budget" as one lump.

=== THE ANCHOR IS NOT REPRODUCIBLE ON THIS DISTRIBUTION -- measured while authoring ===

The autopsy's outcome (1) is worded as an ABSOLUTE bar, "post-hoc R^2 already ~0.99". Measured
on a real-config cell before this was queued (seed 42, b_shipped, `reset_all_rng(42)`, the
exact V3-EXQ-1023 recipe), that bar is UNREACHABLE BY CONSTRUCTION:

    PCA-32 of the P0a buffer, held-out   0.887956
    PCA-32 of the P0a buffer, in-sample  0.892805     <- the achievable ceiling for ANY
    SD-106 code, held-out                0.846929        32-dim linear code on this data
    SD-106 code, in-sample               0.852143
    preservation_holdout.r2 (SGD head)   0.7490045300  <- reproduces V3-EXQ-1023 EXACTLY

PCA-32 explains 0.893 of `world_obs` here, not the design doc's 0.9984 -- and held-out and
in-sample agree to within 0.006, so this is NOT a train/test-split artefact. The design-time
anchor was measured on a DIFFERENT observation distribution, which is a third explanation the
autopsy's three outcomes do not name. It is recorded as a first-class finding
(`diagnostics.anchor_not_reproducible_on_p0a_distribution`) for governance.

=== WHY THE ROUTING IS A DECOMPOSITION, AND NOT A BAR ===

The first draft routed on a bar TRANSCRIBED from the design doc: parity_ratio :=
posthoc_ols_r2 / pca32_ols_r2 >= 0.99, from that doc's own 0.9974/0.9983. The Step 4.5 red-team
pass showed the transfer is NOT UNIQUE. The equally defensible RESIDUAL transfer -- the ON
recipe discarded (1-0.9974)/(1-0.9983) = 1.529x the anchor's residual -- implies a bar of 0.933
on the P0a buffer, and the two give OPPOSITE routes on the authoring seed (0.9538 clears 0.933,
fails 0.99). Anything in [0.93, 0.99) would have been unattributable. The design-time ceiling
0.9983 has a residual of 0.0017, so no statistic transferred across it is numerically stable.

So NO BAR IS TRANSFERRED. The routing decomposes the shortfall V3-EXQ-1023 ACTUALLY REPORTED,
entirely within this run. Per seed, at b_shipped:

    gap            := pca32_ols_r2 - sgd_head_r2        the whole shortfall to be explained
    metric_share   := (posthoc_ols_r2 - sgd_head_r2) / gap
    budget_share   := (best-budget posthoc_ols_r2 - posthoc_ols_r2) / gap
    residual_share := 1 - metric_share - budget_share

The three sum to 1 by construction, are bounded, are always defined once gap > 0 (its own
precondition), and map one-to-one onto the autopsy's three outcomes. Computed WITHIN a seed and
only then aggregated -- a median of medians can pass or fail on seed re-ordering alone in an
exactly-paired design.

=== THE THREE PRE-REGISTERED OUTCOMES (the autopsy's, as shares) ===

Route = the share that leads the runner-up by DOMINANCE_MARGIN (0.15). At most one can lead by
any positive margin, so the criteria are a partition with an explicit tie branch.

  (1) `metrics_never_comparable` -- metric_share dominant. The reported metric under-read the
      code; mechanism and budget are both fine; the consumer-rung shortfall is a genuine
      OBJECTIVE-CHOICE question -> promote the generic-vs-task-relevant compression /lit-pull.
  (2) `under_budgeted_p0a` -- budget_share dominant. Fix the budget and re-run the acceptance
      measurement. Which coordinate to raise is read off the steps-vs-data contrast above.
  (3) residual_share dominant -- neither explains it. SPLIT by the paired OFF comparison at
      b_shipped, because one "build fix" label would otherwise cover two different builds:
        `mechanism_defect_below_ceiling`  sd106 > off: the term works but undershoots.
        `mechanism_defect_inert_at_dv`    sd106 <= off: the term is not reaching the code at
                                          all, despite a non-zero bypass norm.
  (tie) `no_dominant_explanation` -- no share leads by the margin. A RESULT, not an instrument
      failure: the shortfall has no single dominant cause and the per-seed shares say how it
      splits.

NONE of these is a verdict on the SD-106 CLAIM: this is `EXPERIMENT_PURPOSE = "diagnostic"`,
excluded from governance confidence and conflict scoring, and its job is to route the autopsy's
owed decision.

DISCLOSURE, because it bears on pre-registration: seed 42 was run at the real config while
authoring, as the magnitude check queueing requires. Its numbers are printed above and banked in
the manifest; its shares were metric 0.705 / budget 0.199 / residual 0.096. No threshold was
chosen from them -- `DOMINANCE_MARGIN` is data-free (three shares summing to 1 have a uniform
split of 1/3, so a 0.15 lead means the leader holds at least ~0.43), `SHORTFALL_GAP_FLOOR` and
`CEILING_STABILITY_TOL` are set from V3-EXQ-1023's own cross-seed spread and from the
authoring-cell ceiling drift respectively, and both instrument gates are data-independent.
Seeds 43 and 44 are unseen.

=== WHAT A NULL HERE WOULD AND WOULD NOT MEAN ===

Outcome (3) WOULD mean: at weight 200, on this data distribution, the preservation term does not
buy linear decodability that a closed-form probe can find, at any budget in the swept range.
It WOULD NOT mean the preservation pressure is the wrong lever, and it does NOT license a
re-queue at a different weight -- the autopsy already refused that, and this run sweeps BUDGET,
not WEIGHT. It routes to /implement-substrate on SD-106's own code, not to another letter here.

If the instrument control `pca32_ols_r2` fails its floor, the run is an INSTRUMENT failure and
self-routes `substrate_not_ready_requeue` -- never a substrate verdict.

=== DV-SYMMETRY INVARIANCE (one line per arm, Step 3.5) ===

Every arm's DV is `posthoc_ols_r2`: held-out fraction of `world_obs` variance recoverable by an
affine map from the 32-dim code. Its symmetry group is GL(32) acting on the code plus
translations -- any invertible linear reparameterisation of z leaves it exactly unchanged.

  `sd106` vs `off`: the manipulation (`preservation_weight` 0 -> 200, plus the zero-init linear
      bypass) changes WHICH 32-dim subspace of the 250-dim observation space the code spans. A
      change of spanned subspace is NOT a GL(32) action on a fixed subspace, so the manipulation
      is not invariant under the DV's symmetry group. NOT invariant.
  `b_shipped`/`b_steps600`/`b_steps1200`: the manipulation (Adam steps 12 -> 20 -> 40 epochs)
      changes the learned encoder weights and hence the spanned subspace. NOT invariant.
  `b_designref`: the manipulation (rollout episodes 60 -> 100) changes the training distribution
      the subspace is fitted to, hence the subspace. NOT invariant.
  Readout (a) `sgd_head_r2` is deliberately NOT invariant under GL(32) either -- it depends on
      the decoder head's convergence, which is exactly the quantity whose contribution this run
      isolates. Stated so the (a)-vs-(b) gap is read as a measurement, not as an inconsistency.

=== KNOWN OPEN SUBSTRATE DEFECTS (Step 2.5c) ===

Open `degrading` substrate_queue entries overlapping the modules this driver exercises:
`SD-106` and `SD-018`, both listing `ree_core/latent/zworld_p0.py` + `ree_core/latent/stack.py`,
and `SD-MECH303-THRESHOLD-SOURCING` listing `ree_core/utils/config.py` + `ree_core/agent.py`.
Degrading -> recorded, not blocking. The SD-106 entry is this run's own subject.

The three `corrupting` entries open on all-ON agent modules (`mode-governance-engagement`,
`contextmemory-write-path-addressing-degeneracy`, `sd_blocked_agency_mismatch_floor_calibration`)
are NOT exercised here: this driver CONSTRUCTS a REEAgent but never calls `sense()`,
`select_action()` or `step()` on it. The only code path that runs is `ZWorldP0Trainer` over
`latent_stack.split_encoder`, driven by a `RandomPolicy` against the environment. Stated rather
than inherited from V3-EXQ-1023's scoped user exception, because the narrower fact is true.

=== RED-TEAM RECORD ===
red-team (fable, foreground, /queue-experiment Step 4.5): see the queue entry note for the
verdict and the per-finding dispositions recorded at the end of this file.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from experiments._metrics import (  # noqa: E402
    p0_readiness_gate, P0NotReady, dv_headroom_check,
)
from ree_core.latent.zworld_p0 import ZWorldP0Config, ZWorldP0Trainer  # noqa: E402

import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1023_sd106_bottleneck_preservation_validation as x1023  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1041_sd106_preservation_step_budget_metric_diagnostic"
QUEUE_ID = "V3-EXQ-1041"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-106"]

# Every readiness precondition in this driver is reachable BY CONSTRUCTION or against a FROZEN
# RECORDED control, and the one whose reachability is NOT self-evident is gated explicitly
# rather than exempted: `identity_probe_recovers_unity` needs the OLS system over-determined,
# which `probe_system_overdetermined` measures and floors (see PROBE_OVERDETERMINATION_FLOOR).
# The others: `pca32_is_the_optimal_32dim_linear_code` is Eckart-Young, true for every dataset;
# `capture_buffer_length_exact` and `split_replay_mean_predictor_mse_exact` are exact identities
# of the capture seam; `sd106_bypass_trained_off_zero` and
# `shipped_cell_reproduces_1023_sgd_r2` are anchored to V3-EXQ-1023's OWN recorded values
# (skip norm 4.13-4.37; r2 0.6999-0.7490), both verified reachable on a real-config seed-42 cell
# while authoring (skip 4.3682, repro delta 0.0000); `budget_manipulation_reached_substrate`
# reads n_steps straight off the trainer. No hand-written predicate here is narrower than the
# state it anchors to.
ANCHOR_REACHABILITY_EXEMPT = (
    "every readiness predicate is reachable by construction (Eckart-Young, an exact capture "
    "identity, or a substrate-computed count) or against V3-EXQ-1023's own recorded control "
    "values, verified on a real-config seed-42 cell before queueing; the single predicate with "
    "a non-obvious reachability condition (identity_probe_recovers_unity, which needs an "
    "over-determined OLS system) carries its own measured floor precondition "
    "probe_system_overdetermined rather than an exemption."
)

# ---- imported, never re-defined -------------------------------------------------------
RUNG = x1002.RUNG
RUNG_ID = x1002.RUNG_ID
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE          # 200
SHIPPED_ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES  # 60 -- V3-EXQ-1023's own value
PRESERVATION_WEIGHT = x1023.PRESERVATION_WEIGHT      # 200.0
USE_WORLD_ENCODER_SKIP = x1023.USE_WORLD_ENCODER_SKIP
PROJECTION_DIM = 32                                   # the encoder's own world_dim; asserted below

SEEDS: List[int] = [42, 43, 44]
SEED_MAJORITY = x1002.SEED_MAJORITY                   # 2 of 3

# ---- the budget factor ----------------------------------------------------------------
# (episodes, epochs). `n_steps = (n_train // batch_size) * epochs`, batch_size 64.
BUDGETS: Dict[str, Dict[str, int]] = {
    "b_shipped":    {"episodes": SHIPPED_ZWORLD_P0_EPISODES, "epochs": 12},
    "b_steps600":   {"episodes": SHIPPED_ZWORLD_P0_EPISODES, "epochs": 20},
    "b_steps1200":  {"episodes": SHIPPED_ZWORLD_P0_EPISODES, "epochs": 40},
    "b_designref":  {"episodes": 100,                        "epochs": 12},
}
BUDGET_IDS = ["b_shipped", "b_steps600", "b_steps1200", "b_designref"]
SHIPPED_BUDGET = "b_shipped"

TRACK_SD106 = "sd106"
TRACK_OFF = "off"
TRACKS = [TRACK_SD106, TRACK_OFF]

# ---- PRE-REGISTERED THRESHOLDS (constants; never derived from this run's statistics) ---
# THE BARS ARE CEILING-RELATIVE, NOT ABSOLUTE -- see "THE ANCHOR IS NOT REPRODUCIBLE ON THIS
# DISTRIBUTION" in the docstring. An absolute "post-hoc R^2 >= 0.99" bar, which is the
# autopsy's literal wording, is UNREACHABLE BY CONSTRUCTION on the P0a rollout buffer: PCA-32
# of that buffer explains only 0.8928 of world_obs (measured while authoring), so no 32-dim
# linear code of any kind can reach 0.99 there. The autopsy's INTENT survives intact once the
# bar is expressed the way the design doc's own numbers were: as a RATIO to the PCA-32 anchor
# measured on the SAME data by the SAME probe.
#
# NO BAR IS TRANSFERRED FROM THE DESIGN DOC AT ALL -- see "WHY THE ROUTING IS A DECOMPOSITION"
# in the docstring. A first draft routed on `posthoc_ols_r2 / pca32_ols_r2 >= 0.99`, taken from
# the design doc's own 0.9974/0.9983. The red-team pass showed that transfer is not unique: the
# equally defensible RESIDUAL transfer, (1-0.9974)/(1-0.9983) = 1.529x the anchor's residual,
# implies a bar of 0.933 on the P0a buffer -- and the two give OPPOSITE routes on the authoring
# seed (0.9538 clears 0.933, fails 0.99). The design-time ceiling of 0.9983 is near-degenerate
# (its residual is 0.0017), so no statistic transferred across it is stable.
#
# The routing is therefore a DECOMPOSITION of the shortfall V3-EXQ-1023 actually reported,
# computed entirely within this run, with no external bar of any kind. Per seed, at b_shipped:
#     gap            := pca32_ols_r2 - sgd_head_r2      (the whole shortfall to be explained)
#     metric_share   := (posthoc_ols_r2 - sgd_head_r2) / gap
#     budget_share   := (best-budget posthoc_ols_r2 - posthoc_ols_r2) / gap
#     residual_share := 1 - metric_share - budget_share
# The three sum to 1 by construction, are bounded, are always defined once `gap > 0` (its own
# precondition), and map one-to-one onto the autopsy's three outcomes.
#
# DOMINANCE_MARGIN is data-free: three shares summing to 1 have a uniform split of 1/3 each, so
# requiring the leader to beat the runner-up by 0.15 means it holds at least ~0.43 when the
# other two split evenly -- decisively more than a third. At most ONE share can lead by any
# positive margin, so the criteria are a partition with an explicit tie branch.
DOMINANCE_MARGIN = 0.15
# `gap` must be large enough for the shares to mean anything. 0.02 is ~8x the cross-seed spread
# V3-EXQ-1023 recorded on the SGD metric between its two like-for-like 396-step seeds
# (0.7490 vs 0.7464 = 0.0026). Measured while authoring: gap = 0.1390.
SHORTFALL_GAP_FLOOR = 0.02
# RED-TEAM F5: `budget_share` compares posthoc R^2 across budgets against ONE denominator (the
# b_shipped ceiling), but `b_designref` buffers a different rollout and so has its own PCA-32
# ceiling. If the ceiling itself moved, a "budget effect" could be a denominator effect. The
# ceiling must therefore be stable across budgets for the decomposition to be read as one.
# Measured while authoring: 0.8880 (60-episode buffers) vs 0.8840 (100-episode) = 0.004.
CEILING_STABILITY_TOL = 0.02
# INSTRUMENT GATE 1 -- data-INDEPENDENT. An affine map from world_obs to world_obs is the
# identity and lies inside the OLS family, so this MUST return ~1.0 on any data. It fails only
# if the probe's arithmetic, the split, or the numerics are broken. Replaces the earlier
# absolute PCA-32 floor, which was a property of the DATA rather than of the probe.
IDENTITY_PROBE_FLOOR = 0.999
# REACHABILITY OF INSTRUMENT GATE 1, made explicit rather than assumed. The identity control
# can only return ~1.0 when the OLS system is OVER-determined: with `world_obs` itself as the
# code there are 251 free parameters per output, so a train split smaller than that yields the
# minimum-norm solution instead of a least-squares fit and the identity is NOT recovered
# (measured while authoring: overdetermination -77 -> 0.9895, +98 -> 0.9946, +309 -> exactly
# 1.0). Without this floor the identity gate would be unmeetable-by-construction on a small
# buffer -- the exact defect `validate_experiments`'s anchor-reachability lint exists to catch,
# and the same defect that took out this driver's first-draft absolute PCA-32 floor.
# 250 is n_features itself, i.e. the standard n_train >= 2 x n_parameters rule of thumb for a
# well-conditioned least-squares fit; it is the smallest margin at which the identity control
# was measured to return exactly 1.0. Every real-config cell clears it by ~8-13x
# (overdetermination ~1922 at b_shipped, ~3150 at b_designref).
PROBE_OVERDETERMINATION_FLOOR = 250
# INSTRUMENT GATE 2 -- mathematically grounded. PCA-32 is the optimal 32-dim linear code for
# reconstructing its own input, so no trained 32-dim code can beat it on the train split, and
# held-out it can only do so within estimation noise. A real violation means the split or the
# projection is wrong, not that the encoder beat PCA.
PCA_OPTIMALITY_TOL = 0.01
# REPRODUCTION GATE. |my sgd_head_r2 - V3-EXQ-1023's recorded value| on the shipped ON cells.
# 0.05 absolute, NOT 1e-6: V3-EXQ-1023 ran on linux-x86_64-py3.10-torch2.12.0+cpu and this run
# may land on darwin-arm64, where ~400 Adam steps of float accumulation do not reproduce
# bit-identically (CLAUDE.md "Running the test suite" -- assert upstream of the quantizer). The
# bar is set to catch a CONFIG error (a wrong-recipe cell reads ~0.94, i.e. 0.19 away) while
# surviving cross-machine-class float drift. The exact per-seed delta is recorded regardless.
REPRO_TOL = 0.05
REF_1023_SGD_R2: Dict[int, float] = {
    42: 0.7490045122982752,
    43: 0.7463981728369910,
    44: 0.6998874112632603,
}
REF_1023_SOURCE = ("v3_exq_1023_sd106_bottleneck_preservation_validation_20260912T045319Z_v3"
                   " arm_results[zworld_sd106__*].sd106_preservation_holdout_r2")
# Non-degeneracy: the step-budget manipulation must actually have reached the substrate.
BUDGET_STEP_SPREAD_FLOOR = 500   # n_steps(b_steps1200) - n_steps(b_shipped), worst seed
# The split replay must be EXACT. `mean_predictor_mse` depends on the whole buffer, the
# permutation and n, so an exact match is a strong proof that the replayed buffer and the
# reproduced train/holdout split are the ones the substrate actually used.
SPLIT_REPLAY_REL_TOL = 1e-9

# ---- dry-run schedule ------------------------------------------------------------------
# `run_zworld_p0(dry_run=True)` FORCES batch_size=8, epochs=2, which would collapse every
# budget arm onto one value and hide the manipulation from the smoke test. So the smoke passes
# dry_run=False to the warmup and supplies its own shrunken config, preserving the epochs
# RATIO (12:20:40:12 -> 2:4:8:2) so the dry-run still exercises a real budget contrast.
DRY_RUN_SEEDS = [42]
DRY_RUN_STEPS = 30
DRY_RUN_BATCH = 8
# Both instrument gates are data-independent, so the smoke uses the SAME thresholds the real
# run does -- there is no dry-run relaxation of any gate here. Only the reproduction gate is
# dropped under --dry-run (the shrunken config is not V3-EXQ-1023's config, so its reference
# value does not apply); see the filter in run_experiment.
# Episode counts are sized so the smoke's own train split clears
# PROBE_OVERDETERMINATION_FLOOR: the instrument gates then run at their REAL thresholds in the
# smoke, with no dry-run relaxation of any gate anywhere in this driver.
DRY_RUN_BUDGETS: Dict[str, Dict[str, int]] = {
    "b_shipped":   {"episodes": 28, "epochs": 2},
    "b_steps600":  {"episodes": 28, "epochs": 4},
    "b_steps1200": {"episodes": 28, "epochs": 8},
    "b_designref": {"episodes": 44, "epochs": 2},
}

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x724, x734, x1002, x1023)]
_ZG = ZGoalStreamAccumulator()


def _arm_id(track: str, budget: str) -> str:
    return "%s__%s" % (track, budget)


# --------------------------------------------------------------------------------------
# WARMUP -- the canonical P0a path, invoked in V3-EXQ-1023's exact call order
# --------------------------------------------------------------------------------------
def _capturing_target(sink: List[torch.Tensor]):
    """The SD-018 proximity target, delegated VERBATIM, with a `world_obs` tap on the side.

    `run_zworld_p0` calls `target(obs_dict)` exactly once per buffered step, in order, on the
    same `obs_dict` it takes `world_obs` from -- so appending here captures the trainer's
    buffer BY CONSTRUCTION rather than by replaying the rollout and hoping it matches. (It
    does not: a fresh env built from the same seed after the warmup produced a buffer of the
    right LENGTH but different CONTENT, measured 13% off on the mean-predictor baseline. The
    rollout is reproducible only in situ.) `config`/`target_fn` is the seam
    `zworld_p0_warmup.resolve_target_fn` documents for exactly this kind of caller.

    The RETURNED VALUE is `resource_prox_target(obs_dict)` unchanged, so P0a trains on
    byte-identical targets to V3-EXQ-1023 and the reproduction gate still means what it says.
    Only the recorded `p0a_target` string differs, and it is named honestly.
    """
    from experiments._lib.zworld_p0_warmup import resource_prox_target

    def resource_prox_target_capturing(obs_dict: Dict[str, Any]) -> Optional[float]:
        sink.append(obs_dict["world_state"].float().reshape(-1).clone())
        return resource_prox_target(obs_dict)

    return resource_prox_target_capturing


def _warm(track: str, seed: int, env_kwargs: Dict[str, Any], episodes: int,
          steps_per_episode: int, epochs: int, batch_size: Optional[int],
          sink: List[torch.Tensor]) -> Tuple[Any, Dict[str, Any], Any, float]:
    """Run P0a ONLY, reproducing V3-EXQ-1023's construction order exactly.

    1023 warms inside `arm_cell(seed, ...)` (so the global RNG is `reset_all_rng(seed)` at
    entry) and then does, in this order: build the training env, build the agent, snapshot the
    latent stack, build a SECOND env for the warmup rollout, call `_train_all_on_agent`, whose
    FIRST action -- before any optimiser is constructed -- is `run_zworld_p0(...)`. Every step
    after P0a in that helper is P0b/P1, which this diagnostic does not run and which cannot
    affect `preservation_holdout` (it is computed at the end of `trainer.train()`). Calling
    `run_zworld_p0` directly is therefore the SAME computation, not an approximation of it.
    """
    warm_env = x734._make_env(seed, env_kwargs)
    if track == TRACK_SD106:
        agent = x1023._make_sd106_agent(warm_env)
    else:
        agent = x1002._make_agent(warm_env)
    before = latent_stack_snapshot(agent)
    p0_env = x734._make_env(seed, env_kwargs)

    kw: Dict[str, Any] = {"epochs": int(epochs)}
    if batch_size is not None:
        kw["batch_size"] = int(batch_size)
    cfg = ZWorldP0Config(
        preservation_weight=(float(PRESERVATION_WEIGHT) if track == TRACK_SD106 else 0.0),
        resource_field_weight=0.0,    # 978's OFF arm, carried inside the config
        **kw,
    )
    stats = run_zworld_p0(
        agent, p0_env, seed, int(episodes), int(steps_per_episode),
        policy=RandomPolicy(seed), label="ree_allon rung=%s" % RUNG_ID,
        dry_run=False, config=cfg, target_fn=_capturing_target(sink),
    )
    guard = latent_stack_weight_delta(agent, before)
    _ZG.observe(agent)
    skip = getattr(agent.latent_stack.split_encoder, "world_encoder_skip", None)
    skip_norm = float(skip.weight.detach().norm()) if skip is not None else 0.0
    return agent, stats, guard, skip_norm


# --------------------------------------------------------------------------------------
# THE POST-HOC PROBE
# --------------------------------------------------------------------------------------
def _reproduce_split(n: int, cfg: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The trainer's own train/holdout split, recomputed from the resolved config.

    Mirrors `ZWorldP0Trainer.train()` exactly (`ree_core/latent/zworld_p0.py`): a CPU
    `torch.Generator` seeded with `cfg.seed`, `torch.randperm(n)`, then the same two-line
    `n_train` clamp. Any drift in that formula shows up as a `mean_predictor_mse` mismatch and
    fails the `split_replay_exact` precondition rather than producing a wrong number.
    """
    batch_size = int(cfg["batch_size"])
    holdout_fraction = float(cfg["holdout_fraction"])
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))
    perm = torch.randperm(n, generator=gen)
    n_train = max(int(round((1.0 - holdout_fraction) * n)), batch_size)
    n_train = min(n_train, n - 1) if n > batch_size else n
    return perm[:n_train], perm[n_train:]


def _ols_holdout_r2(x_tr: torch.Tensor, y_tr: torch.Tensor,
                    x_te: torch.Tensor, y_te: torch.Tensor) -> Dict[str, Any]:
    """Held-out R^2 of an OLS AFFINE map x -> y, against the train-split mean predictor.

    The R^2 DEFINITION is copied from the substrate's own `preservation_holdout` block so the
    two readouts are directly comparable: `1 - MSE(pred, y_te) / MSE(train_mean, y_te)`. Only
    the FIT differs -- closed-form optimum here, SGD there -- which is the whole point.

    `driver="gelsd"` is the SVD-based minimum-norm solution: the SD-070 code has a measured
    participation ratio around 8 of 32 dimensions, so the 33x33 Gram matrix is rank-deficient
    and a Cholesky/QR solve would be numerically meaningless. The numerical rank is recorded.
    """
    ones_tr = torch.ones(x_tr.shape[0], 1, dtype=x_tr.dtype)
    ones_te = torch.ones(x_te.shape[0], 1, dtype=x_te.dtype)
    a_tr = torch.cat([x_tr, ones_tr], dim=1)
    a_te = torch.cat([x_te, ones_te], dim=1)
    sol = torch.linalg.lstsq(a_tr.double(), y_tr.double(), driver="gelsd")
    w = sol.solution
    pred_te = a_te.double() @ w
    mse = float(((pred_te - y_te.double()) ** 2).mean())
    base = float(((y_tr.double().mean(dim=0, keepdim=True).expand_as(y_te) - y_te.double())
                  ** 2).mean())
    sv = torch.linalg.svdvals(a_tr.double())
    tol = float(sv.max()) * max(a_tr.shape) * 2.220446049250313e-16
    return {
        "mse": mse,
        "mean_predictor_mse": base,
        "r2": (1.0 - mse / base) if base > 0.0 else None,
        "numerical_rank": int((sv > tol).sum()),
        "n_features": int(a_tr.shape[1]),
        "n_train": int(a_tr.shape[0]),
        "n_holdout": int(a_te.shape[0]),
    }


def _pca_project(obs_tr: torch.Tensor, obs_te: torch.Tensor,
                 dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCA-`dim` of `world_obs`, fitted on the TRAIN split only, applied to both splits.

    This is the object the SD-106 design doc calls the PCA-32 anchor: "PCA-32 of the encoder's
    own 250-dim input at the encoder's own 32-dim width", recorded there at R^2 0.9983/0.9984.
    """
    mu = obs_tr.double().mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(obs_tr.double() - mu, full_matrices=False)
    comp = vh[:int(dim)].T
    return ((obs_tr.double() - mu) @ comp).float(), ((obs_te.double() - mu) @ comp).float()


def _z_world(agent: Any, cfg_obj: ZWorldP0Config, obs: torch.Tensor) -> torch.Tensor:
    """The trained z_world code for a batch of observations, via the SUBSTRATE's own path.

    `ZWorldP0Trainer.__init__` constructs no module and consumes no RNG (it only binds the
    latent stack and empties its buffers), so building a probe trainer is inert. Reaching for
    `_z_world_path` rather than re-deriving `world_encoder + skip` inline is deliberate: a
    re-derivation would silently go stale if the substrate's world path changes, which is
    exactly the class of error this whole diagnostic exists to catch elsewhere.
    """
    if not hasattr(ZWorldP0Trainer, "_z_world_path"):
        raise RuntimeError(
            "ZWorldP0Trainer._z_world_path is gone. The post-hoc probe MUST use the "
            "substrate's own world path; refusing to re-derive it inline."
        )
    probe = ZWorldP0Trainer(agent.latent_stack, cfg_obj)
    with torch.no_grad():
        return probe._z_world_path(obs)


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def _config_slice(track: str, budget: str, seed: int, sched: Dict[str, Any]) -> Dict[str, Any]:
    b = sched["budgets"][budget]
    return {
        "arm_id": _arm_id(track, budget),
        "arm_track": track,
        "arm_budget": budget,
        "rung_id": RUNG_ID,
        "env_kwargs": sched["env_kwargs"],
        "zworld_p0_episodes": int(b["episodes"]),
        "zworld_p0_epochs": int(b["epochs"]),
        "zworld_p0_batch_size": sched["batch_size"],
        "steps_per_episode": int(sched["steps"]),
        # The SD-106 manipulation belongs IN the slice: two cells differing in it are not the
        # same computation, and omitting it would collide the ON and OFF fingerprints.
        "preservation_weight": (float(PRESERVATION_WEIGHT) if track == TRACK_SD106 else 0.0),
        "use_world_encoder_skip": bool(USE_WORLD_ENCODER_SKIP and track == TRACK_SD106),
        "resource_field_weight": 0.0,
        "projection_dim": int(PROJECTION_DIM),
    }


def run_cell(track: str, budget: str, seed: int, sched: Dict[str, Any]) -> Dict[str, Any]:
    arm = _arm_id(track, budget)
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm), flush=True)
    b = sched["budgets"][budget]
    slice_ = _config_slice(track, budget, seed, sched)

    with arm_cell(seed, config_slice=slice_, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS) as cell:
        sink: List[torch.Tensor] = []
        agent, stats, guard, skip_norm = _warm(
            track, seed, sched["env_kwargs"], b["episodes"], sched["steps"], b["epochs"],
            sched["batch_size"], sink)

        row: Dict[str, Any] = {
            "arm_id": arm, "arm_track": track, "arm_budget": budget, "seed": int(seed),
            "p0a_ran": bool(stats.get("p0a_ran")),
            "p0a_reason": stats.get("p0a_reason"),
            "n_buffered": stats.get("p0a_n_buffered"),
            "n_steps": stats.get("p0a_n_steps"),
            "final_loss": stats.get("p0a_final_loss"),
            "mean_loss": stats.get("p0a_mean_loss"),
            "used_preservation_head": stats.get("p0a_used_preservation_head"),
            "used_world_encoder_skip": stats.get("p0a_used_world_encoder_skip"),
            "skip_weight_norm": float(skip_norm),
            "world_encoder_weight_delta": guard,
            "grounding_label_balance": stats.get("p0a_grounding_label_balance"),
            "holdout_mean_lift": stats.get("p0a_holdout_mean_lift"),
            "p0a_holdout": stats.get("p0a_holdout"),
            "p0a_config": stats.get("p0a_config"),
        }
        pres = stats.get("p0a_preservation_holdout")
        row["sgd_head_r2"] = pres.get("r2") if isinstance(pres, dict) else None
        row["sgd_head_mse"] = pres.get("mse") if isinstance(pres, dict) else None
        row["sgd_head_mean_predictor_mse"] = (pres.get("mean_predictor_mse")
                                              if isinstance(pres, dict) else None)

        if not row["p0a_ran"]:
            row["posthoc_ols_r2"] = None
            row["pca32_ols_r2"] = None
            row["parity_ratio"] = None
            row["identity_probe_r2"] = None
            row["probe_overdetermination"] = None
            row["code_minus_pca32"] = None
            row["split_replay_rel_delta"] = None
            row["capture_n_delta"] = None
            row["capture_n_delta_abs"] = None
            cell.stamp(row)
            print("verdict: FAIL", flush=True)
            return row

        # ---- the post-hoc probe, on the SAME buffer and the SAME split ----------------
        cfg_dict = dict(stats.get("p0a_config") or {})
        obs = torch.stack(sink)
        row["capture_n_delta"] = int(obs.shape[0]) - int(row["n_buffered"] or 0)
        # RED-TEAM F6: the gate takes the worst ABSOLUTE delta. Gating on the max SIGNED delta
        # and then abs()-ing it lets a negative cell hide behind a zero one.
        row["capture_n_delta_abs"] = abs(int(row["capture_n_delta"]))
        row["world_obs_dim"] = int(obs.shape[1])

        tr_idx, te_idx = _reproduce_split(int(obs.shape[0]), cfg_dict)
        obs_tr, obs_te = obs[tr_idx], obs[te_idx]
        # Computed EXACTLY as the substrate computes its own `mean_predictor_mse`
        # (`F.mse_loss` over float32, not a float64 re-derivation), so a mismatch means the
        # SPLIT differs rather than the arithmetic precision.
        base_capture = float(torch.nn.functional.mse_loss(
            obs_tr.mean(dim=0, keepdim=True).expand_as(obs_te), obs_te))
        row["capture_mean_predictor_mse"] = base_capture

        ref_base = row["sgd_head_mean_predictor_mse"]
        if ref_base is not None and float(ref_base) > 0.0:
            row["split_replay_rel_delta"] = abs(base_capture - float(ref_base)) / float(ref_base)
        else:
            # The OFF track builds no `_preserve_head`, so the substrate reports no
            # `mean_predictor_mse` to compare against. `capture_n_delta` is the check that
            # applies to every cell; this one is scoped to cells that can carry it.
            row["split_replay_rel_delta"] = None

        cfg_obj = ZWorldP0Config(**cfg_dict)
        z_tr = _z_world(agent, cfg_obj, obs_tr)
        z_te = _z_world(agent, cfg_obj, obs_te)
        if int(z_tr.shape[1]) != int(PROJECTION_DIM):
            # The PCA anchor is "PCA-32 at the ENCODER'S OWN width"; if the encoder's width is
            # not 32 the anchor is measuring a different object and the comparison is void.
            # Crash rather than quietly compare unlike widths.
            raise RuntimeError(
                "z_world width is %d but PROJECTION_DIM is %d; the PCA anchor would no longer "
                "be at the encoder's own width. Refusing to run."
                % (int(z_tr.shape[1]), int(PROJECTION_DIM)))
        row["z_dim"] = int(z_tr.shape[1])
        row["z_participation_ratio"] = x1002._participation_ratio(z_tr)
        row["z_all_finite"] = bool(torch.isfinite(z_tr).all() and torch.isfinite(z_te).all())

        ols = _ols_holdout_r2(z_tr, obs_tr, z_te, obs_te)
        row["posthoc_ols"] = ols
        row["posthoc_ols_r2"] = ols["r2"]
        # In-sample sibling, recorded because the design doc does not say which of the two its
        # 0.9974/0.9983 table reports and the pair separates "the fit overfits" from "the data
        # is different". (Measured while authoring on seed 42: they differ by <0.006, so the
        # design-doc gap is NOT a train/test-split artefact.)
        row["insample_ols_r2"] = _ols_holdout_r2(z_tr, obs_tr, z_tr, obs_tr)["r2"]

        p_tr, p_te = _pca_project(obs_tr, obs_te, PROJECTION_DIM)
        pca = _ols_holdout_r2(p_tr, obs_tr, p_te, obs_te)
        row["pca32_ols"] = pca
        row["pca32_ols_r2"] = pca["r2"]
        row["pca32_insample_ols_r2"] = _ols_holdout_r2(p_tr, obs_tr, p_tr, obs_tr)["r2"]

        # INSTRUMENT GATE 1: an affine map world_obs -> world_obs is the identity and is inside
        # the OLS family, so this is ~1.0 on ANY data -- PROVIDED the system is over-determined.
        ident = _ols_holdout_r2(obs_tr, obs_tr, obs_te, obs_te)
        row["identity_probe_r2"] = ident["r2"]
        row["probe_overdetermination"] = int(ident["n_train"]) - int(ident["n_features"])
        # INSTRUMENT GATE 2: PCA-32 is the optimal 32-dim linear code for its own input.
        row["code_minus_pca32"] = (
            None if (row["posthoc_ols_r2"] is None or row["pca32_ols_r2"] is None)
            else float(row["posthoc_ols_r2"]) - float(row["pca32_ols_r2"]))
        # THE CEILING-RELATIVE DV: how much of the ACHIEVABLE 32-dim linear decodability this
        # code actually realises, measured on the same data by the same probe. This is the
        # quantity the design doc's own 0.9974/0.9983 pair expresses, and the one every
        # criterion routes on.
        row["parity_ratio"] = (
            None if (row["posthoc_ols_r2"] is None or not row["pca32_ols_r2"])
            else float(row["posthoc_ols_r2"]) / float(row["pca32_ols_r2"]))

        # The gap this diagnostic exists to size.
        row["posthoc_minus_sgd"] = (
            None if (row["posthoc_ols_r2"] is None or row["sgd_head_r2"] is None)
            else float(row["posthoc_ols_r2"]) - float(row["sgd_head_r2"]))
        row["posthoc_minus_pca32"] = (
            None if (row["posthoc_ols_r2"] is None or row["pca32_ols_r2"] is None)
            else float(row["posthoc_ols_r2"]) - float(row["pca32_ols_r2"]))
        row["repro_1023_delta"] = (
            abs(float(row["sgd_head_r2"]) - REF_1023_SGD_R2[int(seed)])
            if (track == TRACK_SD106 and budget == SHIPPED_BUDGET
                and row["sgd_head_r2"] is not None and int(seed) in REF_1023_SGD_R2)
            else None)

        cell.stamp(row)

    print("  [cell] %s seed=%d n_steps=%s n_buf=%s sgd_r2=%s ols_r2=%s pca32=%s parity=%s"
          % (arm, seed, row["n_steps"], row["n_buffered"],
             _fmt(row["sgd_head_r2"]), _fmt(row["posthoc_ols_r2"]),
             _fmt(row["pca32_ols_r2"]), _fmt(row["parity_ratio"])),
          flush=True)
    print("verdict: %s" % ("PASS" if row["posthoc_ols_r2"] is not None else "FAIL"), flush=True)
    return row


def _fmt(v: Optional[float]) -> str:
    return "None" if v is None else ("%.4f" % float(v))


def _flat_scalar(d: Dict[str, Any]) -> Dict[str, Any]:
    """The flat scalar projection the runpack converter and the indexer actually read.

    Booleans are excluded by `_is_number` downstream, so every flag in here is already an
    int; non-finite and None values are DROPPED rather than emitted, because a nan IS numeric
    to the indexer and would pollute a delta, while an absent key correctly reads as
    unmeasured. Standard sec 3b "Machine-readable verdict readout".
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, bool):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or f in (float("inf"), float("-inf")):
            continue
        out[k] = int(v) if isinstance(v, int) else f
    return out


# --------------------------------------------------------------------------------------
# AGGREGATION + ROUTING
# --------------------------------------------------------------------------------------
def _worst(rows: List[Dict[str, Any]], key: str, mode: str) -> Tuple[Optional[float],
                                                                     Optional[str]]:
    """Worst cell for `key`. `mode` is 'min' (floors) or 'max' (ceilings). Returns the
    extremum AND the offending cell id, never a mean -- the indexer recomputes `met` from
    the reported number, so an in-band mean would mask an out-of-band cell."""
    vals = [(float(r[key]), "%s|seed%d" % (r["arm_id"], r["seed"]))
            for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    return (min(vals) if mode == "min" else max(vals))


def _median(rows: List[Dict[str, Any]], track: str, budget: str,
            key: str) -> Optional[float]:
    vals = [float(r[key]) for r in rows
            if r["arm_track"] == track and r["arm_budget"] == budget
            and r.get(key) is not None]
    return statistics.median(vals) if vals else None


def _cell(rows: List[Dict[str, Any]], track: str, budget: str,
          seed: int) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r["arm_track"] == track and r["arm_budget"] == budget and r["seed"] == seed:
            return r
    return None


def _seed_shares(rows: List[Dict[str, Any]], seed: int) -> Optional[Dict[str, Any]]:
    """The shortfall decomposition for ONE seed -- PAIRED, never a median of medians.

    RED-TEAM F4: taking medians of `posthoc_ols_r2` first and differencing afterwards can pass
    or fail on seed RE-ORDERING alone in an exactly-paired design (shipped {0.94,0.95,0.96} vs
    best {0.99,0.96,0.97} medians rise 0.02 while the per-seed rises are {0.05,0.01,0.01}).
    The shares are therefore computed WITHIN a seed and only then aggregated.
    """
    ship = _cell(rows, TRACK_SD106, SHIPPED_BUDGET, seed)
    if ship is None:
        return None
    sgd, post, ceil_ = (ship.get("sgd_head_r2"), ship.get("posthoc_ols_r2"),
                        ship.get("pca32_ols_r2"))
    if sgd is None or post is None or ceil_ is None:
        return None
    best = None
    best_budget = None
    for b in BUDGET_IDS:
        c = _cell(rows, TRACK_SD106, b, seed)
        v = None if c is None else c.get("posthoc_ols_r2")
        if v is not None and (best is None or float(v) > best):
            best, best_budget = float(v), b
    if best is None:
        return None
    gap = float(ceil_) - float(sgd)
    off = _cell(rows, TRACK_OFF, SHIPPED_BUDGET, seed)
    off_post = None if off is None else off.get("posthoc_ols_r2")
    out: Dict[str, Any] = {
        "seed": int(seed), "gap": gap, "sgd_head_r2": float(sgd),
        "posthoc_ols_r2_shipped": float(post), "pca32_ols_r2_shipped": float(ceil_),
        "posthoc_ols_r2_best": best, "best_budget": best_budget,
        "off_posthoc_ols_r2_shipped": (None if off_post is None else float(off_post)),
        "sd106_minus_off_shipped": (None if off_post is None
                                    else float(post) - float(off_post)),
    }
    if gap > 0.0:
        out["metric_share"] = (float(post) - float(sgd)) / gap
        out["budget_share"] = (best - float(post)) / gap
        out["residual_share"] = 1.0 - out["metric_share"] - out["budget_share"]
    else:
        out["metric_share"] = out["budget_share"] = out["residual_share"] = None
    return out


def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    sched = {
        "budgets": DRY_RUN_BUDGETS if dry_run else BUDGETS,
        "steps": DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE,
        "batch_size": DRY_RUN_BATCH if dry_run else None,
        "env_kwargs": env_kwargs,
    }

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for budget in BUDGET_IDS:
            for track in TRACKS:
                rows.append(run_cell(track, budget, seed, sched))

    sd_rows = [r for r in rows if r["arm_track"] == TRACK_SD106]
    shipped_on = [r for r in sd_rows if r["arm_budget"] == SHIPPED_BUDGET]

    # ---- preconditions ---------------------------------------------------------------
    capture_n_worst, capture_n_cell = _worst(rows, "capture_n_delta_abs", "max")
    split_worst, split_cell = _worst(
        [r for r in rows if r.get("split_replay_rel_delta") is not None],
        "split_replay_rel_delta", "max")
    pca_worst, pca_cell = _worst(rows, "pca32_ols_r2", "min")
    ident_worst, ident_cell = _worst(rows, "identity_probe_r2", "min")
    overdet_worst, overdet_cell = _worst(rows, "probe_overdetermination", "min")
    optim_worst, optim_cell = _worst(rows, "code_minus_pca32", "max")
    skip_worst, skip_cell = _worst(sd_rows, "skip_weight_norm", "min")
    repro_worst, repro_cell = _worst(shipped_on, "repro_1023_delta", "max")

    step_spread: List[float] = []
    for seed in seeds:
        lo = [r["n_steps"] for r in sd_rows
              if r["seed"] == seed and r["arm_budget"] == SHIPPED_BUDGET and r["n_steps"]]
        hi = [r["n_steps"] for r in sd_rows
              if r["seed"] == seed and r["arm_budget"] == "b_steps1200" and r["n_steps"]]
        if lo and hi:
            step_spread.append(float(hi[0] - lo[0]))
    step_spread_worst = min(step_spread) if step_spread else None

    shares = [sh for sh in (_seed_shares(rows, sd) for sd in seeds) if sh is not None]
    gap_worst = min((sh["gap"] for sh in shares), default=None)
    # RED-TEAM F5: one denominator is used for every share, so the ceiling must not move.
    ceil_drift = None
    ceil_drift_cell = None
    for sd in seeds:
        base = _cell(rows, TRACK_SD106, SHIPPED_BUDGET, sd)
        if base is None or base.get("pca32_ols_r2") is None:
            continue
        for b in BUDGET_IDS:
            c = _cell(rows, TRACK_SD106, b, sd)
            if c is None or c.get("pca32_ols_r2") is None:
                continue
            d = abs(float(c["pca32_ols_r2"]) - float(base["pca32_ols_r2"]))
            if ceil_drift is None or d > ceil_drift:
                ceil_drift, ceil_drift_cell = d, "%s|seed%d" % (c["arm_id"], sd)

    checks = [
        {"name": "capture_buffer_length_exact", "kind": "readiness",
         "measured": (capture_n_worst if capture_n_worst is not None else 1e9),
         "threshold": 0.0, "direction": "upper",
         "control": ("`run_zworld_p0` calls the target fn exactly once per buffered step, so a "
                     "tap on it has length delta exactly 0 by construction"),
         "offending_cell": capture_n_cell,
         "description": ("The post-hoc probe's captured world_obs buffer must be the same "
                         "length the substrate reported buffering. A non-zero delta means the "
                         "capture seam moved and the probe is reading a different dataset "
                         "than the one that trained.")},
        {"name": "split_replay_mean_predictor_mse_exact", "kind": "readiness",
         "measured": (split_worst if split_worst is not None else 1e9),
         "threshold": SPLIT_REPLAY_REL_TOL, "direction": "upper",
         "control": ("`mean_predictor_mse` reported by the substrate's own "
                     "preservation_holdout block on the SD-106 cells -- it is a function of "
                     "the whole buffer, the permutation and n_train, so an exact match proves "
                     "the reproduced split is the substrate's"),
         "offending_cell": split_cell,
         "description": ("The reproduced train/holdout split must equal the trainer's. Scoped "
                         "to SD-106 cells: the OFF track builds no preservation head and "
                         "therefore reports no reference base to compare against; "
                         "capture_buffer_length_exact is the check that covers every cell.")},
        {"name": "probe_system_overdetermined", "kind": "readiness",
         "measured": (overdet_worst if overdet_worst is not None else -1e9),
         "threshold": float(PROBE_OVERDETERMINATION_FLOOR), "direction": "lower",
         "control": ("n_train minus the OLS design-matrix width (world_obs_dim + 1) on the "
                     "identity control; positive means least-squares rather than minimum-norm"),
         "offending_cell": overdet_cell,
         "description": ("REACHABILITY of the identity gate below, measured rather than "
                         "assumed. Under-determined, the identity control cannot return unity "
                         "on ANY correct implementation, so gating on it would be unmeetable "
                         "by construction.")},
        {"name": "identity_probe_recovers_unity", "kind": "readiness",
         "measured": (ident_worst if ident_worst is not None else 0.0),
         "threshold": IDENTITY_PROBE_FLOOR, "direction": "lower",
         "control": ("world_obs -> world_obs: the identity map is inside the OLS family, so "
                     "this returns ~1.0 on ANY data, on any machine, at any budget"),
         "offending_cell": ident_cell,
         "description": ("INSTRUMENT GATE 1, data-independent. If the probe cannot recover "
                         "unity from the identity control, a low posthoc_ols_r2 on the trained "
                         "code is the probe's arithmetic failing, not the encoder's, and no "
                         "outcome is attributable. Deliberately NOT an absolute PCA-32 floor: "
                         "the PCA-32 level is a property of the DATA (0.893 on the P0a buffer "
                         "vs the design doc's 0.9984 on some other distribution), so a floor "
                         "on it would gate the science on the dataset rather than the probe.")},
        {"name": "pca32_is_the_optimal_32dim_linear_code", "kind": "readiness",
         "measured": (optim_worst if optim_worst is not None else 1e9),
         "threshold": PCA_OPTIMALITY_TOL, "direction": "upper",
         "control": ("PCA-32 minimises reconstruction MSE over all rank-32 linear codes of its "
                     "own input (Eckart-Young), so no trained 32-dim code can exceed it beyond "
                     "held-out estimation noise"),
         "offending_cell": optim_cell,
         "description": ("INSTRUMENT GATE 2, mathematically grounded. A trained code beating "
                         "the PCA-32 anchor by more than the tolerance means the split or the "
                         "projection is wrong, not that the encoder beat PCA.")},
        {"name": "sd106_bypass_trained_off_zero", "kind": "readiness",
         "measured": (skip_worst if skip_worst is not None else 0.0),
         "threshold": 1e-9, "direction": "lower", "comparator": ">",
         "control": ("the bypass is zero-initialised, so a norm above zero is proof P0a "
                     "actually stepped it; V3-EXQ-1023 measured 4.13-4.37"),
         "offending_cell": skip_cell,
         "description": ("The SD-106 manipulation must have happened. A zero norm means the "
                         "flag read as enabled while the mechanism stayed inert.")},
        {"name": "shipped_cell_reproduces_1023_sgd_r2", "kind": "readiness",
         "measured": (repro_worst if repro_worst is not None else 1e9),
         "threshold": REPRO_TOL, "direction": "upper",
         "control": REF_1023_SOURCE,
         "offending_cell": repro_cell,
         "description": ("REPRODUCTION GATE. The b_shipped SD-106 cells re-run V3-EXQ-1023's "
                         "exact P0a config, so their sgd_head_r2 must land on 1023's recorded "
                         "0.6999-0.7490. Tolerance 0.05 absolute, not machine-epsilon: 1023 "
                         "ran on linux-x86_64/torch2.12 and ~400 Adam steps do not reproduce "
                         "bit-identically across machine classes. A config error reads ~0.94, "
                         "0.19 away, which this bar still catches.")},
        dv_headroom_check(
            "share_margin_headroom",
            dv_name="share_margin (leading share minus runner-up)",
            criterion_threshold=DOMINANCE_MARGIN,
            achievable=1.0,
            statistic="analytic_bound",
            margin=2.0,
            control="analytic: the three shares sum to 1 and are bounded, so the leading "
                    "share's margin over the runner-up spans [-1, 1]; a 0.15 lead sits at "
                    "0.15 of the achievable 1.0, i.e. well inside it, on ANY dataset",
            description=("DV headroom for the load-bearing criteria. The margin's reachable "
                         "range is a property of the decomposition, not of this run's data, "
                         "which is exactly why the routing was moved off a transferred bar. "
                         "The EMPIRICAL half of feasibility -- that the gap the shares divide "
                         "is itself non-degenerate -- is carried by shortfall_gap_positive "
                         "below, measured rather than assumed."),
        ),
        {"name": "shortfall_gap_positive", "kind": "readiness",
         "measured": (gap_worst if gap_worst is not None else -1e9),
         "threshold": SHORTFALL_GAP_FLOOR, "direction": "lower",
         "control": ("pca32_ols_r2 minus sgd_head_r2 at b_shipped -- the shortfall V3-EXQ-1023 "
                     "reported, measured against the in-run achievable ceiling; 0.1390 on the "
                     "authoring seed"),
         "description": ("The three explanatory shares are fractions of this gap. With a gap at "
                         "or below zero they are undefined or unstable, and no route is "
                         "attributable.")},
        {"name": "ceiling_stable_across_budgets", "kind": "readiness",
         "measured": (ceil_drift if ceil_drift is not None else 1e9),
         "threshold": CEILING_STABILITY_TOL, "direction": "upper",
         "control": ("pca32_ols_r2 per budget against the b_shipped value on the same seed; "
                     "0.004 on the authoring seed (0.8880 at 60 episodes, 0.8840 at 100)"),
         "offending_cell": ceil_drift_cell,
         "description": ("RED-TEAM F5. Every share uses the b_shipped ceiling as its single "
                         "denominator, so a budget that moved the ceiling itself could present "
                         "a denominator effect as a budget effect.")},
        {"name": "budget_manipulation_reached_substrate", "kind": "readiness",
         "measured": (step_spread_worst if step_spread_worst is not None else 0.0),
         "threshold": float(BUDGET_STEP_SPREAD_FLOOR if not dry_run else 5),
         "direction": "lower",
         "control": ("n_steps is computed by the substrate as (n_train // batch_size) * "
                     "epochs, so the delta is read off the trainer, not asserted"),
         "description": ("Non-degeneracy for the budget factor: b_steps1200 must actually have "
                         "run many more optimiser steps than b_shipped at the same data.")},
    ]

    if dry_run:
        # The smoke runs a shrunken config, so the b_shipped cells are NOT V3-EXQ-1023's
        # cells and the reproduction reference is meaningless against them. Drop the check
        # rather than fake a threshold for it; it is unconditional on every real run.
        checks = [c for c in checks
                  if c["name"] != "shipped_cell_reproduces_1023_sgd_r2"]

    gate_green = True
    gate_reason = None
    try:
        preconditions = p0_readiness_gate(checks)
    except P0NotReady as exc:
        gate_green = False
        preconditions = exc.preconditions
        gate_reason = "; ".join(
            str(p.get("name")) for p in preconditions if not p.get("met", True))

    # ---- criteria + routing ----------------------------------------------------------
    r_by_budget = {b: _median(rows, TRACK_SD106, b, "parity_ratio") for b in BUDGET_IDS}
    off_by_budget = {b: _median(rows, TRACK_OFF, b, "parity_ratio") for b in BUDGET_IDS}
    abs_by_budget = {b: _median(rows, TRACK_SD106, b, "posthoc_ols_r2") for b in BUDGET_IDS}
    pca_by_budget = {b: _median(rows, TRACK_SD106, b, "pca32_ols_r2") for b in BUDGET_IDS}

    def _med(key: str) -> Optional[float]:
        vals = [float(sh[key]) for sh in shares if sh.get(key) is not None]
        return statistics.median(vals) if vals else None

    m_share, b_share, r_share = _med("metric_share"), _med("budget_share"), _med("residual_share")
    sd_minus_off = _med("sd106_minus_off_shipped")
    defined = [v for v in (m_share, b_share, r_share) if v is not None]

    def _margin(v: Optional[float], others: List[Optional[float]]) -> Optional[float]:
        o = [x for x in others if x is not None]
        return None if (v is None or not o) else float(v) - max(o)

    m_marg = _margin(m_share, [b_share, r_share])
    b_marg = _margin(b_share, [m_share, r_share])
    r_marg = _margin(r_share, [m_share, b_share])

    c1 = bool(m_marg is not None and m_marg >= DOMINANCE_MARGIN)
    c2 = bool(b_marg is not None and b_marg >= DOMINANCE_MARGIN)
    c3 = bool(r_marg is not None and r_marg >= DOMINANCE_MARGIN)

    if not gate_green:
        label = "substrate_not_ready_requeue"
    elif len(defined) < 3:
        label = "substrate_not_ready_requeue"
        gate_green = False
        gate_reason = (gate_reason or "the shortfall decomposition was undefined on a seed")
    elif c1:
        label = "metrics_never_comparable"
    elif c2:
        label = "under_budgeted_p0a"
    elif c3:
        # RED-TEAM F2: the OFF track entered no criterion, so one "build fix" route covered both
        # an INERT preservation term and a WORKING one that undershoots -- two different builds.
        # The paired OFF comparison at b_shipped separates them.
        label = ("mechanism_defect_inert_at_dv"
                 if (sd_minus_off is not None and float(sd_minus_off) <= 0.0)
                 else "mechanism_defect_below_ceiling")
    else:
        label = "no_dominant_explanation"

    criteria = [
        {"name": "C1_metric_share_dominant", "load_bearing": True, "passed": c1,
         "measured": m_marg, "threshold": DOMINANCE_MARGIN, "comparator": ">=",
         "metric_share_median": m_share,
         "detail": ("median over seeds of (posthoc_ols_r2 - sgd_head_r2) / gap at b_shipped, "
                    "minus the larger of the other two shares. Routes outcome (1) "
                    "metrics_never_comparable: the reported metric under-read the code")},
        {"name": "C2_budget_share_dominant", "load_bearing": True, "passed": c2,
         "measured": b_marg, "threshold": DOMINANCE_MARGIN, "comparator": ">=",
         "budget_share_median": b_share,
         "detail": ("median over seeds of (best-budget posthoc_ols_r2 - b_shipped "
                    "posthoc_ols_r2) / gap, minus the larger of the other two shares. Routes "
                    "outcome (2) under_budgeted_p0a")},
        {"name": "C3_residual_share_dominant", "load_bearing": True, "passed": c3,
         "measured": r_marg, "threshold": DOMINANCE_MARGIN, "comparator": ">=",
         "residual_share_median": r_share,
         "detail": ("median over seeds of 1 - metric_share - budget_share, minus the larger of "
                    "the other two shares: what neither the metric nor the budget explains. "
                    "Routes outcome (3), split by the paired OFF comparison into "
                    "mechanism_defect_inert_at_dv (sd106 <= off) and "
                    "mechanism_defect_below_ceiling (sd106 > off)")},
    ]
    combination_rule = (
        "SHORTFALL DECOMPOSITION, not an AND/OR of PASSes. Per seed at b_shipped: "
        "gap = pca32_ols_r2 - sgd_head_r2; metric_share = (posthoc_ols_r2 - sgd_head_r2)/gap; "
        "budget_share = (best-budget posthoc_ols_r2 - posthoc_ols_r2)/gap; residual_share = "
        "1 - metric_share - budget_share. The three sum to 1 by construction, so AT MOST ONE "
        "can lead the runner-up by any positive margin and the criteria are a partition. The "
        "route is the leader when its margin reaches DOMINANCE_MARGIN (0.15), else "
        "`no_dominant_explanation` -- which is a RESULT, not an instrument failure: it says the "
        "shortfall has no single dominant cause, and the recorded shares say how it splits. A "
        "failed precondition overrides everything and self-routes substrate_not_ready_requeue. "
        "`outcome` is PASS when the diagnostic DISCRIMINATED (gate green and a share dominant), "
        "FAIL when it did not -- never a verdict on SD-106, which this run does not score."
    )
    criteria_non_degenerate = {
        # Every share is a bounded fraction of an in-run, in-run-denominated gap, so all three
        # are reachable whenever the probe works (identity gate) and the gap is positive. That
        # is the property the earlier transferred bar did NOT have.
        "C1_metric_share_dominant": bool(
            ident_worst is not None and float(ident_worst) >= IDENTITY_PROBE_FLOOR
            and gap_worst is not None and float(gap_worst) >= SHORTFALL_GAP_FLOOR),
        # C2 additionally needs the budget factor to have actually varied.
        "C2_budget_share_dominant": bool(
            ident_worst is not None and float(ident_worst) >= IDENTITY_PROBE_FLOOR
            and gap_worst is not None and float(gap_worst) >= SHORTFALL_GAP_FLOOR
            and step_spread_worst is not None
            and float(step_spread_worst) >= (BUDGET_STEP_SPREAD_FLOOR if not dry_run else 5)),
        "C3_residual_share_dominant": bool(
            ident_worst is not None and float(ident_worst) >= IDENTITY_PROBE_FLOOR
            and gap_worst is not None and float(gap_worst) >= SHORTFALL_GAP_FLOOR
            and step_spread_worst is not None),
    }
    non_degenerate = bool(any(criteria_non_degenerate.values()))

    outcome = "PASS" if (gate_green and label not in ("substrate_not_ready_requeue",
                                                     "no_dominant_explanation")) else "FAIL"

    readout: Dict[str, Any] = {
        "n_seeds": len(seeds),
        "n_cells": len(rows),
        "gate_green": 1 if gate_green else 0,
        "non_degenerate": 1 if non_degenerate else 0,
        "dominance_margin": DOMINANCE_MARGIN,
        "shortfall_gap_floor": SHORTFALL_GAP_FLOOR,
        "ceiling_stability_tol": CEILING_STABILITY_TOL,
        "metric_share_median": m_share,
        "budget_share_median": b_share,
        "residual_share_median": r_share,
        "metric_share_margin": m_marg,
        "budget_share_margin": b_marg,
        "residual_share_margin": r_marg,
        "shortfall_gap_worst": gap_worst,
        "ceiling_drift_worst": ceil_drift,
        "sd106_minus_off_posthoc_shipped_median": sd_minus_off,
        "identity_probe_floor": IDENTITY_PROBE_FLOOR,
        "identity_probe_r2_worst": ident_worst,
        "probe_overdetermination_worst": overdet_worst,
        "code_minus_pca32_worst": optim_worst,
        "pca32_ols_r2_worst": pca_worst,
        "repro_1023_delta_worst": repro_worst,
        "sd106_skip_weight_norm_worst": skip_worst,
        "budget_step_spread_worst": step_spread_worst,
        "c1_metric_share_dominant": 1 if c1 else 0,
        "c2_budget_share_dominant": 1 if c2 else 0,
        "c3_residual_share_dominant": 1 if c3 else 0,
        "design_doc_pca32_anchor": 0.9983,
        "design_doc_on_recipe_obs_r2": 0.9974,
    }
    for b in BUDGET_IDS:
        if r_by_budget[b] is not None:
            readout["sd106_parity_ratio_median__%s" % b] = float(r_by_budget[b])
        if off_by_budget[b] is not None:
            readout["off_parity_ratio_median__%s" % b] = float(off_by_budget[b])
        if abs_by_budget[b] is not None:
            readout["sd106_posthoc_ols_r2_median__%s" % b] = float(abs_by_budget[b])
        if pca_by_budget[b] is not None:
            readout["pca32_ols_r2_median__%s" % b] = float(pca_by_budget[b])
        sgd = [float(r["sgd_head_r2"]) for r in sd_rows
               if r["arm_budget"] == b and r.get("sgd_head_r2") is not None]
        if sgd:
            readout["sd106_sgd_head_r2_median__%s" % b] = statistics.median(sgd)

    gap = [float(r["posthoc_minus_sgd"]) for r in shipped_on
           if r.get("posthoc_minus_sgd") is not None]
    if gap:
        readout["shipped_posthoc_minus_sgd_median"] = statistics.median(gap)
    readout = _flat_scalar(readout)

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE,
                               datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "evidence_direction": "unknown",
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate
                              else "no criterion could discriminate: %s" % (gate_reason,)),
        "outcome": outcome,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "sleep_driver_pattern": "none",
        "dry_run": bool(dry_run),
        "combination_rule": combination_rule,
        "criteria": criteria,
        "readout": readout,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria": criteria,
            "combination_rule": combination_rule,
            "criteria_non_degenerate": criteria_non_degenerate,
            "gate_reason": gate_reason,
            "shares": {"metric": m_share, "budget": b_share, "residual": r_share,
                       "margins": {"metric": m_marg, "budget": b_marg, "residual": r_marg},
                       "sd106_minus_off_posthoc_shipped": sd_minus_off},
            "routes": {
                "metrics_never_comparable": (
                    "The code is already at PCA-32 parity under a post-hoc probe, so the "
                    "in-training SGD head metric and the design-time post-hoc probe were never "
                    "the same statistic. Mechanism and budget are both fine. Promote the "
                    "generic-vs-task-relevant compression /lit-pull to primary; the consumer-"
                    "rung shortfall V3-EXQ-1023 measured is an OBJECTIVE-CHOICE question."),
                "under_budgeted_p0a": (
                    "P0a is under-budgeted. Raise the budget and re-run the acceptance "
                    "measurement. Read the steps-vs-data contrast (b_steps600 vs b_designref "
                    "at ~equal steps, b_shipped/600/1200 at equal data) to say WHICH."),
                "mechanism_defect_below_ceiling": (
                    "Neither the metric nor the budget explains the shortfall, and the "
                    "preservation term DOES beat the paired OFF arm at b_shipped -- so the "
                    "mechanism works but undershoots the achievable ceiling across a 3.3x step "
                    "range at fixed data AND a 1.7x data range at fixed steps. Route to "
                    "/implement-substrate on the preservation leg's own code. Does NOT license "
                    "a re-queue at another preservation_weight."),
                "mechanism_defect_inert_at_dv": (
                    "As above, but the SD-106 arm does NOT beat the paired OFF arm at "
                    "b_shipped: the preservation term is INERT at the DV despite a non-zero "
                    "bypass norm. A different build from the one above -- the term is not "
                    "reaching the code, rather than reaching it and undershooting."),
                "no_dominant_explanation": (
                    "No share leads the runner-up by DOMINANCE_MARGIN. This is a RESULT, not an "
                    "instrument failure: the shortfall has no single dominant cause and the "
                    "recorded per-seed shares say how it splits. Governance should read the "
                    "shares rather than treat the run as unrun."),
                "substrate_not_ready_requeue": (
                    "A precondition failed. No outcome is attributable; this is an instrument "
                    "result, not a verdict on SD-106."),
            },
            "residual_ambiguity": (
                "The swept budget range is 3.3x in optimiser steps and 1.7x in observations. A "
                "flat reading is evidence against a budget explanation WITHIN that range and "
                "is not a proof about arbitrarily larger budgets; the design doc's own named "
                "reference point (600 steps, 4000 observations) IS inside the range, which is "
                "what makes outcome (3) actionable rather than merely unresolved."),
        },
        "diagnostics": {
            "per_seed_shortfall_decomposition": shares,
            "sd106_parity_ratio_median_by_budget": r_by_budget,
            "off_parity_ratio_median_by_budget": off_by_budget,
            "sd106_posthoc_ols_r2_median_by_budget": abs_by_budget,
            "pca32_ols_r2_median_by_budget": pca_by_budget,
            "anchor_not_reproducible_on_p0a_distribution": {
                "finding": (
                    "PCA-32 of the P0a rollout buffer explains ~0.89 of world_obs (held-out "
                    "AND in-sample, so it is not a split artefact), against the SD-106 design "
                    "doc's 0.9983/0.9984 for 'PCA-32 of the encoder's own 250-dim input'. The "
                    "design-time anchor is therefore NOT on this observation distribution, "
                    "which is a third explanation the autopsy's three outcomes do not name. "
                    "Every criterion here is consequently CEILING-RELATIVE (parity_ratio), so "
                    "the routing is unaffected by it -- but governance should see it, because "
                    "it means the absolute 0.9974 figure SD-106 was designed against cannot be "
                    "compared with any number measured on a P0a buffer."),
                "measured_while_authoring_seed42": {
                    "pca32_heldout_r2": 0.887956, "pca32_insample_r2": 0.892805,
                    "code_heldout_r2": 0.846929, "code_insample_r2": 0.852143,
                    "sgd_head_r2": 0.7490045300303669,
                    "sd106_parity_by_budget": {"b_shipped": 0.9538, "b_steps600": 0.9677,
                                               "b_steps1200": 0.9850, "b_designref": 0.9673},
                    "off_parity_by_budget": {"b_shipped": 0.9245, "b_steps600": 0.9259,
                                             "b_steps1200": 0.9182, "b_designref": 0.9303},
                    "note": ("recorded here as the authoring-time magnitude check required "
                             "before queueing (seed 42 only; 43 and 44 unseen). The run "
                             "re-measures all of them across all three seeds."),
                },
            },
            "repro_1023_reference": REF_1023_SGD_R2,
            "repro_1023_source": REF_1023_SOURCE,
            "design_doc_reference": {
                "shipped_obs_r2": [0.9432, 0.9493],
                "preservation_200_skip_obs_r2": [0.9974, 0.9978],
                "pca32_anchor_obs_r2": [0.9983, 0.9984],
                "stated_realistic_budget": "600 steps, 4000 buffered observations",
                "note": ("the design doc does NOT record the budget behind its own R^2 table; "
                         "b_designref reconstructs the budget it names elsewhere"),
            },
        },
    }

    full_config = {
        "rung_id": RUNG_ID,
        "env_kwargs": env_kwargs,
        "seeds": list(seeds),
        "tracks": list(TRACKS),
        "budgets": sched["budgets"],
        "steps_per_episode": sched["steps"],
        "batch_size_override": sched["batch_size"],
        "preservation_weight": float(PRESERVATION_WEIGHT),
        "use_world_encoder_skip": bool(USE_WORLD_ENCODER_SKIP),
        "projection_dim": int(PROJECTION_DIM),
        "dominance_margin": DOMINANCE_MARGIN,
        "shortfall_gap_floor": SHORTFALL_GAP_FLOOR,
        "ceiling_stability_tol": CEILING_STABILITY_TOL,
        "identity_probe_floor": IDENTITY_PROBE_FLOOR,
        "pca_optimality_tol": PCA_OPTIMALITY_TOL,
        "probe_overdetermination_floor": PROBE_OVERDETERMINATION_FLOOR,
        "repro_tol": REPRO_TOL,
        "budget_step_spread_floor": BUDGET_STEP_SPREAD_FLOOR,
        "split_replay_rel_tol": SPLIT_REPLAY_REL_TOL,
        "seed_majority": int(SEED_MAJORITY),
    }
    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    manifest["_out_path"] = str(out_path)
    return manifest


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(seeds, dry_run=args.dry_run)

    print("")
    print("=" * 78)
    print("%s -- %s" % (QUEUE_ID, result["interpretation"]["label"]))
    print("outcome: %s  non_degenerate: %s" % (result["outcome"], result["non_degenerate"]))
    for c in result["criteria"]:
        print("  %-46s passed=%-5s measured=%s threshold=%s"
              % (c["name"], c["passed"], _fmt(c["measured"]), c["threshold"]))
    for p in result["interpretation"]["preconditions"]:
        print("  [precond] %-46s met=%-5s measured=%s threshold=%s"
              % (p.get("name"), p.get("met"), _fmt(p.get("measured")), p.get("threshold")))
    print("manifest: %s" % result["_out_path"])
    print("=" * 78)

    if args.dry_run:
        # Smoke assertions -- the decisive readout must be non-trivially engaged BEFORE the
        # full grid is committed to (Step 3.5 "even for evidence-purpose scripts").
        rows = result["arm_results"]
        assert all(r["p0a_ran"] for r in rows), "a P0a cell refused its buffer in the smoke"
        assert all(r.get("posthoc_ols_r2") is not None for r in rows), \
            "posthoc_ols_r2 is None on some cell -- the decisive readout did not compute"
        assert all(r.get("pca32_ols_r2") is not None for r in rows), \
            "pca32_ols_r2 is None on some cell -- the anchor did not compute"
        assert all(int(r["probe_overdetermination"]) >= PROBE_OVERDETERMINATION_FLOOR
                   for r in rows), \
            "OLS probe under-determined: %s" % [r["probe_overdetermination"] for r in rows]
        assert all(float(r["identity_probe_r2"]) >= IDENTITY_PROBE_FLOOR for r in rows), \
            "identity probe below unity: %s" % [r["identity_probe_r2"] for r in rows]
        assert all(float(r["code_minus_pca32"]) <= PCA_OPTIMALITY_TOL for r in rows), \
            "a trained code beat PCA-32: %s" % [r["code_minus_pca32"] for r in rows]
        assert all(int(r["capture_n_delta"]) == 0 for r in rows), \
            "buffer capture length mismatch: %s" % [r["capture_n_delta"] for r in rows]
        assert all(float(r["split_replay_rel_delta"]) <= SPLIT_REPLAY_REL_TOL
                   for r in rows if r.get("split_replay_rel_delta") is not None), \
            "split reproduction inexact: %s" % [r["split_replay_rel_delta"] for r in rows]
        sd = [r for r in rows if r["arm_track"] == TRACK_SD106]
        assert all(float(r["skip_weight_norm"]) > 0.0 for r in sd), \
            "SD-106 bypass never trained off zero in the smoke"
        assert all(r.get("sgd_head_r2") is not None for r in sd), \
            "SD-106 cells reported no preservation_holdout -- the ON arm is not an ON arm"
        off = [r for r in rows if r["arm_track"] == TRACK_OFF]
        assert all(r.get("sgd_head_r2") is None for r in off), \
            "an OFF cell built a preservation head -- the OFF arm is not an OFF arm"
        # The budget manipulation must MOVE the DV's own driver (n_steps) across levels.
        steps = sorted({int(r["n_steps"]) for r in sd})
        assert len(steps) >= 3, "budget arms did not vary n_steps: %s" % (steps,)
        print("[smoke] all assertions passed; n_steps across sd106 budgets: %s" % (steps,))

    return result, args


if __name__ == "__main__":
    _result, _args = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_result["_out_path"],
        dry_run=_args.dry_run,
    )
