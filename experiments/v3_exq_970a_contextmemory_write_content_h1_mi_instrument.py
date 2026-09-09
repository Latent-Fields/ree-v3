#!/opt/local/bin/python3
"""
V3-EXQ-970a -- ContextMemory write-content discrimination: H1 (loss-objective-
mismatch) contrastive-loss leg RE-RUN ON THE REDESIGNED MUTUAL-INFORMATION
INSTRUMENT. Supersedes V3-EXQ-970.

Chip: chip-20260906-ctxmem-instrument-redesign-970a
Campaign: chip-20260907-campaign-w4-s3-contextmemory-content (S3 item 1)

experiment_purpose: diagnostic

hypothesis_space:
  qid: contextmemory_write_content_discrimination
  hid: H1-loss-objective-mismatch

sleep_driver_pattern: N/A (no sleep loop)

red-team (fable), 2026-09-07: CONTESTED -> 5 findings + 3 notes, all FIXED;
see the RED-TEAM section at the end of this docstring.

WHY THIS RUN EXISTS -- the user's 2026-09-06 decision (substrate entry
`contextmemory-write-path-addressing-degeneracy`, field decision_2026_09_06):
the CONTENT half of the ContextMemory write path cannot be decided, only
measured, and the frozen question's live gate is "instrument redesign, not
another leg". The 2026-09-03 cluster autopsy
(failure_autopsy_contextmemory-write-content-cluster_2026-09-03) established
that all four H1-H4 legs failed to adjudicate for ONE structural reason: the
2-cluster occupied-set Jaccard DV is near-binary (72 of 89 portfolio cells
could only read 0.0 or 1.0), so paired tests tied (969: 8/10 seeds at delta
0), the exact sign-flip p-grid floored above alpha (970 Regime B: 0.0625 vs
0.025), and 20x-apart gain settings were indistinguishable. V3-EXQ-970's own
Regime A never produced its DV at all: its held-out split needed > 208
dangerous states and every cell realised 171-195 (dangerous episodes end in
~3.8 steps). Regime B reached the DV's best value on the clusters it trained
on (Jaccard 0.0, 6/6) while its FRESH-cluster readout was bit-identical
between untrained and trained arms -- memorisation without transfer.

This driver executes the substrate entry's implementation_hint (1) VERBATIM,
and nothing beyond it:

  1. INSTRUMENT: the 2-cluster occupied-SET Jaccard is replaced by a
     MUTUAL-INFORMATION statistic over the FULL (cluster x slot) contingency
     table. DV = NMI_excess = NMI_raw - mean(NMI under label shuffle), where
     NMI_raw = I(cluster; slot) / H(cluster) in bits (the fraction of class
     entropy the written slot explains; 1.0 iff the slot determines the
     class, 0.0 iff independent). The label-shuffle null (`MI_N_SHUFFLE`
     permutations of the class labels over the SAME slot sequence) removes
     the finite-sample upward bias of a plug-in MI estimate, which at
     n ~ 70 draws over a 2 x 16 table is ~0.15 bits -- large relative to the
     effects in play, so an uncorrected estimate would read structure into
     noise. The statistic is CONTINUOUS wherever a cluster's draws spread
     over >= 2 slots (see dv_headroom below), unlike set-Jaccard.
  2. PROBE_CLUSTERS >= 4 on the synthetic instrument (K = PROBE_CLUSTERS).
  3. The per-draw contingency table is RECORDED in the manifest for every
     probe of every cell (`contingency` fields, K x NUM_SLOTS integer counts),
     so any later reader can recompute MI, NMI, a different normalisation,
     or a different null without re-running.
  4. The FRESH-cluster (generalisation) readout is the LOAD-BEARING DV; the
     trained-set readout is recorded as SECONDARY only. In Regime B "fresh"
     = a disjoint draw of K cluster bases (seed + 500_000) the arm never
     trained on; in Regime A it = a genuinely HELD-OUT split of real latents
     collected AFTER training, with every optimiser stopped (see 5).
  5. The held-out split is sized as a FRACTION of the realised class count,
     never a fixed N: Regime A appends HELDOUT_EPISODES = 40% of
     TRAINING_EPISODES collection episodes (alternating contexts) in which NO
     objective is stepped -- not E1/E2, not the H1 loss, not the substrate's
     addressing loss -- so the held-out latents are unseen by EVERY arm's
     training uniformly (970's pool-exclusion trick only protected them from
     the H1 sampler; the substrate's own addressing loss trains on E1's
     sequence buffer, which 970's held-out tail sat inside). The probe is
     then BALANCED to min(n_safe, n_dangerous), class-INTERLEAVED, and
     STEP-MATCHED (both classes restricted to within-episode steps 1..
     STEP_MATCH_MAX -- red-team F5, see below). A cell whose balanced count
     is < MIN_HELDOUT_PER_CLASS contributes no load-bearing value (its seed
     is dropped from the paired test, readout still recorded); the unit's
     precondition `heldout_adequate_pairs` then requires >= MIN_PAIRED_SEEDS
     adequate pairs so the sign-flip p-floor still clears alpha.
  6. OBJECTIVE CONVERGENCE is asserted as a P0-kind precondition on every
     TRAINED arm: the arm's own objective, monitored across training, must
     decrease by >= CONV_MIN_DECREASE_FRAC of its analytic range between the
     first and last 5% of the trajectory (971's coupling loss ROSE across
     training and its null was read as a mechanism result). Worst cell over
     the arm's seeds is what is reported (worst-cell rule).
  7. dv_headroom is DECLARED (kind "dv_headroom", built with
     experiments/_metrics.dv_headroom_check, statistic "ceiling_headroom"
     on the UNTRAINED control arm's realised NMI_excess against the DV's
     analytic ceiling 1.0) and the DV's achievable range was MEASURED AT
     PROBE SCALE BEFORE the bar was pre-registered -- see "AUTHORING-TIME
     HEADROOM MEASUREMENT" below, which is where PROBE_CLUSTERS, PROBE_JITTER
     and NMI_MARGIN come from.
  8. Selection mode: every learned arm runs contextmemory_write_selection=
     'gumbel_learned' with its tagger shaped by REAL SGD (the entry: flag-
     flipping alone exercises only the occupancy effect). The DIVERSITY_956
     arm does so through the substrate's own path, contextmemory_write_
     addressing_loss_weight = 0.5 via agent.compute_prediction_loss(); the
     H1_CONTRASTIVE arm through the content-referencing contrastive loss
     this driver injects (weight LAMBDA_H1 = 0.5, addressing weight 0.0 so
     the two objectives never mix within an arm).
  9. An UNTRAINED, SEED-MATCHED baseline arm (same init, same seed, tagger
     never receives a gradient) -- 969's cross-seed-set baseline was the
     defect.
 10. A 'refractory' k=2 arm as the INTERIM-ENABLEMENT REFERENCE: what the
     fleet's mandated interim write-selection rule delivers on this same
     instrument. Descriptive, no test.

This is ONE experiment (the H1 leg on the new instrument), not the four-leg
portfolio. The autopsy decides from its result whether H2-H4 re-run.

INTERIM ENABLEMENT (decision_2026_09_06, DECLARED BY HAND): the WARN-only lint
chip-20260906-ctxmem-enablement-lint (HK-B item 7) had NOT landed on origin/main
when this driver was authored (2026-09-07; chip open and unclaimed), so the
rule is declared here rather than by the linter. This driver is a content-
discrimination experiment -- the one class for which 'gumbel_learned' is
reserved -- and every gumbel_learned arm that trains does so with real SGD
on write_addr_tagger (DIVERSITY_956: contextmemory_write_addressing_loss_weight
0.5 through the substrate path; H1_CONTRASTIVE: the injected contrastive
loss). REFRACTORY_K2 (contextmemory_write_selection='refractory',
contextmemory_write_refractory_k=2) is the mandated interim rule itself and
is run as the reference arm. UNTRAINED is the seed-matched null the spec
requires (gumbel_learned, tagger frozen) -- a control, not a production
configuration. The library default ('argmin') is not touched.

ARMS (identical set in both regimes; arm x seed cells, RNG fully reset at
each cell entry via arm_cell):
  UNTRAINED       gumbel_learned; LAMBDA_H1 = 0; addressing weight 0.
                  write_addr_tagger frozen at init (seed-matched null).
  H1_CONTRASTIVE  gumbel_learned; contrastive loss (LAMBDA_H1 = 0.5) through
                  real SGD; addressing weight 0.  [the H1 manipulation]
  DIVERSITY_956   gumbel_learned; the substrate's own content-BLIND pairwise-
                  diversity addressing loss, weight 0.5 (V3-EXQ-956's exact
                  operating point), through real SGD; LAMBDA_H1 = 0.
                  [H1's discriminating partner: H1 says a content-referencing
                  objective is REQUIRED, so this arm should NOT raise the DV]
  REFRACTORY_K2   refractory, k = 2; no tagger, no addressing objective.
                  [interim-enablement reference, descriptive]

REGIME A (real agent; 956/970's harness, reused): TRAINING_EPISODES = 100 x
STEPS_PER_EPISODE = 150 alternating safe/dangerous contexts every
CONTEXT_SWITCH_EVERY = 5 episodes (block parity), E1/E2 trained every step
as in 956, harm_eval trained as in 956, write-stream state tensor captured
at steps 1..STEP_MATCH_MAX of every episode into per-context buffers (exact
ContextMemory.write() call-site construction, cat([z_self, z_world])). The
H1 loss is injected EVERY step (H1_INJECT_EVERY_N_STEPS = 1) on a balanced
batch from the two buffers, so its tagger update count matches the
DIVERSITY_956 arm's every-step substrate loss (red-team F3). Every
gumbel_learned arm draws the same batch indices each step (UNTRAINED then
discards them) so the three are RNG-matched up to the tagger gradient. Then
HELDOUT_EPISODES = 40 collection episodes with the agent in eval mode and NO
optimiser step: those latents are the held-out split. LOAD-BEARING readout: NMI_excess(context; slot) on the balanced,
interleaved held-out real latents (K = 2 real contexts; H(class) = 1 bit).
SECONDARY: the synthetic K-cluster FRESH probe (same instrument as Regime
B's load-bearing readout) on the same trained tagger -- cross-regime
generalisation, descriptive.

REGIME B (synthetic, no environment): direct SGD on write_addr_tagger alone,
N_STEPS_B = 15000 (V3-EXQ-907's schedule, as 970), on a K = PROBE_CLUSTERS
cluster stream (bases at CLUSTER_BASE_SCALE, per-draw jitter PROBE_JITTER,
continuous generator). H1_CONTRASTIVE minimises the K-class generalisation
of the contrastive loss (mean over cluster pairs of -L1 between class-mean
selection distributions; range [-2, 0]); DIVERSITY_956 minimises
ContextMemory.compute_write_addressing_loss on the same stream (range [0, 1]).
LOAD-BEARING readout: NMI_excess on K FRESH cluster bases (seed + 500_000).
SECONDARY: NMI_excess on the TRAINED bases with fresh noise (970's F1
readout -- kept so a reader sees the memorisation/transfer gap directly).
REFRACTORY_K2 and UNTRAINED run the probes only.

STATISTICS. For each regime R and each trained arm T in {H1_CONTRASTIVE,
DIVERSITY_956}: paired per-seed diffs d_i = NMI_excess(T, i) -
NMI_excess(UNTRAINED, i) on the load-bearing readout; pass(R, T) iff the
(R, T) gate is green AND mean(d) >= NMI_MARGIN AND the exact paired sign-flip
permutation p-value < ALPHA_CORRECTED = 0.05 / 4 (2 regimes x 2 trained
arms). SEEDS has 8 entries so the attainable p-floor is 1/2**8 = 0.0039 and
one tied pair (1/2**7 = 0.0078) still clears 0.0125 -- 970's n=6 fix held
only with zero ties, and the DV was tied by construction. NMI_excess is
continuous at the pre-registered operating point (see headroom measurement),
so ties are not expected; the extra seeds are the margin.

LOAD-BEARING CRITERION `H1_contrastive_raises_generalising_content_conditioning
_in_either_regime` (load_bearing: true): pass(A, H1) OR pass(B, H1).
combination_rule: OR across regimes (either regime succeeding is evidence
for H1's directional prediction; the null requires both to fail their own
gated comparison). SECONDARY criterion `H1_content_reference_required`: in
every regime where H1 passes, DIVERSITY_956 does NOT pass -- H1's necessity
sub-claim, recorded with its own combination rule, not gating.

VERDICT GRID (interpretation.label):
  both (R,H1) gates red                         -> instrument_gate_not_ready
  H1 passes in >= 1 regime, DIV passes in none
    of those regimes                            -> h1_confirmed_content_reference_required[_both_regimes|_real_agent_only|_synthetic_only]
  H1 passes and DIV also passes in that regime  -> training_raises_content_conditioning_content_reference_not_required
  H1 fails, DIV passes                          -> h1_inverted_content_blind_objective_suffices
  neither passes, >= 1 gate green               -> h1_not_confirmed_on_mi_instrument
A synthetic-only positive carries the same H4/SD-070 caveat 970 recorded
(points at the input distribution; the residual ~10x update-budget asymmetry
between regimes is unchanged from 970).

AUTHORING-TIME HEADROOM MEASUREMENT (2026-09-07, scratchpad probe
ctxmem_headroom_probe.py, seeds {1..6} DISJOINT from SEEDS, N = 1500 draws,
untrained gumbel_learned taggers, refractory k=2 and argmin rules, plus a
1500-step design-time positive control of each trained objective probed on
FRESH clusters). Numbers are NMI_excess; the values that fixed the constants
are recorded here so the pre-registration is auditable:
  K=4  jitter 0.0078: untrained 0.328+-0.169 (spread_frac 0.25, QUANTISED); refractory 0.208; argmin 0.116;
       H1-trained fresh 0.572, DIV-trained fresh 0.612; probe repeat-noise SD 0.005
  K=4  jitter 0.030 : untrained 0.207+-0.196 (spread_frac 0.54); H1 fresh 0.489; DIV fresh 0.547; noise SD 0.007
  K=4  jitter 0.078 : untrained 0.112+-0.099 (spread_frac 0.88); refractory 0.069; argmin 0.068;
       H1 fresh 0.226+-0.066; DIV fresh 0.295+-0.082; noise SD 0.007
  K=8  jitter 0.0078: untrained 0.208+-0.158 (spread_frac 0.15, QUANTISED); H1 fresh 0.632; DIV fresh 0.592
  K=8  jitter 0.030 : untrained 0.157+-0.126 (spread_frac 0.46); H1 fresh 0.521; DIV fresh 0.482
  K=8  jitter 0.078 : untrained 0.078+-0.060 (spread_frac 0.92, CONTINUOUS); refractory 0.042; argmin 0.035;
       H1 fresh 0.260+-0.036 (paired diffs vs untrained all >= +0.09); DIV fresh 0.294+-0.029;
       trained-set readouts 0.96-0.98 in every trained cell; probe repeat-noise SD 0.006;
       both objectives converged (H1 -0.016 -> -1.998; DIV 0.999 -> 0.10)
Chosen operating point: PROBE_CLUSTERS = 8, PROBE_JITTER = 0.078
(ratio 1.0 to CLUSTER_BASE_SCALE -- the smallest jitter at which the
statistic is continuous rather than quantised; the legacy 0.0078 leaves each
cluster routed to ONE slot so NMI collapses onto the 5-level collision count,
the same aliasing the autopsy diagnosed in set-Jaccard). NMI_MARGIN = 0.10 = max(absolute floor 0.10,
3 x the probe's repeat-noise SD at this operating point (0.006)).
Both trained objectives moved the fresh-cluster DV at design time, so the
instrument resolves graded change; whether the CONTENT-REFERENCING one is
REQUIRED is exactly what the DIVERSITY_956 arm discriminates at run time.

evidence_direction: "non_contributory". claim_ids: [] -- tests a substrate
mechanism's objective design, not a claim hypothesis (956/970/972's own
convention); excluded from governance confidence scoring by design. Does
NOT flip the substrate entry's status regardless of outcome (decision_2026_
09_06: implemented_pending_validation / corrupting stand until the
instrument-redesign run is ADJUDICATED by governance, not by this driver).

Step 2.4 GOV-REUSE-1: the decisive readout (shuffle-corrected NMI over the
cluster x slot contingency table, fresh/held-out, for a contrastively-
trained tagger) is recorded in NO manifest -- 970 recorded occupied-SET
Jaccard only (no contingency tables; the per-cluster occupied sets it did
record collapse the counts), 956/969/971/972 likewise. Not recoverable;
run fresh.
Step 2.5b re-derive brake: N/A (claim_ids = []); and the autopsy's own
Step 5 records the brake released by user decision for this lineage.
Step 2.5c substrate-path overlap (re-checked 2026-09-07 against the current
substrate_queue.json): the only open `corrupting` entry overlapping a module
this driver exercises is contextmemory-write-path-addressing-degeneracy
(e1_deep.py::ContextMemory.write) -- the entry UNDER TEST, not an unrelated
overlap. mode-governance-engagement / SD-082 / SD-e1-rollout-consistency-
training removed by construction exactly as 970 documents (no salience
coordinator, no lateral-PFC analog, no predict_long_horizon).

SUBSTRATE PROPERTIES held constant across all cells (956/970 verbatim):
contextmemory_gated_content_write=True; sd016_writepath_mode="sense_only";
alpha_world=0.9; use_noise_floor=True; context_memory.memory.requires_grad_
(False) after construction (436e/f Adam-drift neutralisation) -- write_addr_
tagger is the manipulated parameter and stays trainable; use_per_stream_vs /
use_anchor_sets / use_sd039_anchor_payload True; salience coordinator,
coalition controller, sd016_enabled at REEConfig defaults (False).

ethics_preflight: all involvement flags false; decision: allow (V3
pre-ethical instrumentation; SENT-0).

RED-TEAM (fable, 2026-09-07, one pass, Step 4.5): CONTESTED -- 5 findings,
3 notes; every finding verified against the source and FIXED in this file
(no re-spawn: none was BLOCKING, per the one-pass rule).
  F1 a gate-red DIVERSITY_956 control was read as "does not pass" and hence
     as confirming the necessity sub-claim -> `required_evaluable` now needs
     the control's gate GREEN in every H1-passing regime; otherwise the label
     is `..._necessity_unscored_*` and the criterion reads evaluable: false.
  F2 `instrument_gate_not_ready` fired only when all four units were red ->
     it now fires when NEITHER H1 unit is scorable; top-level non_degenerate
     and the load-bearing criterion's non-degeneracy come from the H1 units
     only; interpretation.preconditions carries the H1 units' own entries
     in that case so the indexer sees which gate vacated the arm.
  F3 within-Regime-A budget asymmetry (~100 vs ~1500 tagger updates) ->
     H1 injected every step; update counts matched; RNG-matched draws.
  F4 DIVERSITY_956's convergence gate monitored a balanced cross-context
     batch, not the trained objective's consecutive-window form ->
     `_substrate_addressing_loss_on_window` reproduces agent.py's batch
     construction under no_grad and feeds the gate; the balanced value is
     kept as `objective_trajectory_balanced_batch` (descriptive).
  F5 "context" aliased with time-since-reset (dangerous episodes end in ~4
     steps; z_self EMA-attenuated post-reset) -> STEP_MATCH_MAX = 4 step-
     matching of BOTH classes in training pools and held-out; per-class
     z_self / z_world norms recorded in every held-out readout as the audit
     (smoke: safe/dangerous z_self 0.36/0.32 after matching).
  N1 the `not_required` label lost the regime pattern -> pattern suffixed.
  N2 power at ~38 held-out draws per class -> HELDOUT_EPISODES 20 -> 40.
  N3 write count included held-out writes; whole-unit held-out floor let one
     short cell red both A units -> training-phase count recorded; per-cell
     adequacy + `heldout_adequate_pairs` (>= 7) replaces the min-over-cells
     floor.
Cleared by the reviewer (not raised): eval-mode selection is pure tagger
argmin (no conscience bias); UNTRAINED's tagger genuinely frozen; H1
gradient reaches only the tagger; probe ordering cannot contaminate the
load-bearing readouts.
"""

import argparse
import itertools
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest, flat_readout  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec, evaluate_arm_gate, aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
)
from experiments._metrics import dv_headroom_check  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_970a_contextmemory_write_content_h1_mi_instrument"
QUEUE_ID = "V3-EXQ-970a"
SUPERSEDES = "V3-EXQ-970"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# 970's six plus two. Attainable exact sign-flip p-floor 1/2**8 = 0.0039;
# one tied pair still gives 1/2**7 = 0.0078 < ALPHA_CORRECTED (see docstring).
SEEDS: List[int] = [42, 7, 13, 100, 200, 300, 400, 500]

LAMBDA_H1 = 0.5                       # 970/956/907's weight convention
WRITE_ADDRESSING_LOSS_WEIGHT_956 = 0.5  # V3-EXQ-956's exact operating point
REFRACTORY_K = 2                      # decision_2026_09_06 interim rule

# Arm table -- identical in both regimes. `trained` marks arms whose own
# objective is monitored for the convergence precondition.
ARMS: List[Dict[str, Any]] = [
    {"name": "UNTRAINED", "write_selection": "gumbel_learned",
     "lambda_h1": 0.0, "waddr_w": 0.0, "trained": False, "objective": None},
    {"name": "H1_CONTRASTIVE", "write_selection": "gumbel_learned",
     "lambda_h1": LAMBDA_H1, "waddr_w": 0.0, "trained": True,
     "objective": "contrastive"},
    {"name": "DIVERSITY_956", "write_selection": "gumbel_learned",
     "lambda_h1": 0.0, "waddr_w": WRITE_ADDRESSING_LOSS_WEIGHT_956,
     "trained": True, "objective": "addressing_diversity"},
    {"name": "REFRACTORY_K2", "write_selection": "refractory",
     "lambda_h1": 0.0, "waddr_w": 0.0, "trained": False, "objective": None},
]
TRAINED_ARMS = [a["name"] for a in ARMS if a["trained"]]
BASELINE_ARM = "UNTRAINED"
REFERENCE_ARM = "REFRACTORY_K2"

# Analytic range of each trained objective (for the convergence fraction).
OBJECTIVE_RANGE = {"contrastive": 2.0, "addressing_diversity": 1.0}
CONV_MIN_DECREASE_FRAC = 0.02   # decrease >= 2% of the objective's range
CONV_EDGE_FRAC = 0.05           # first/last 5% of the monitored trajectory

# Substrate properties held constant (956/970 verbatim).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 100
HELDOUT_FRAC = 0.40
HELDOUT_EPISODES = int(round(TRAINING_EPISODES * HELDOUT_FRAC))  # 40 (red-team N2: power)
# red-team F5: "context" in this harness is aliased with time-since-reset
# (dangerous episodes end in ~3.8 steps; z_self is EMA-attenuated to 0.30/0.51/
# 0.66/0.76 of steady state at steps 1-4 after agent.reset()). Both classes'
# training pools AND held-out probe sets are therefore restricted to states
# with within-episode step index < STEP_MATCH_MAX, so the two classes share
# the same post-reset step distribution and the only systematic difference
# between them is the env context. Per-class latent norms are recorded as the
# audit for this (heldout_real_readout.class_latent_norms).
STEP_MATCH_MAX = 4
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000
MAX_STATE_BUF = 4000

MIN_H1_BUF = 8
H1_BATCH_PER_SIDE = 32
# red-team F3: the substrate addressing loss (DIVERSITY_956) steps the tagger
# on EVERY training step via agent.compute_prediction_loss(); 970's every-10-
# steps injection gave the H1 arm ~15x fewer tagger updates inside Regime A
# and confounded the within-regime contrast with budget. Cadence 1 matches
# the two trained arms' update counts (~1500/cell each; the between-regime
# ~10x gap to N_STEPS_B is unchanged from 970 and still declared).
H1_INJECT_EVERY_N_STEPS = 1

MIN_HELDOUT_PER_CLASS = 24     # per-CELL adequacy of the balanced held-out probe
# red-team N3: a whole-unit min-over-cells floor lets ONE short cell red both
# Regime A units. Instead an inadequate cell's load-bearing readout is set to
# None (dropped from the paired test, as 970's _paired_diffs already did) and
# the unit's precondition is the number of ADEQUATE PAIRS: >= 7 keeps the exact
# sign-flip p-floor (1/2**7 = 0.0078) below ALPHA_CORRECTED.
MIN_PAIRED_SEEDS = 7
# PreconditionSpec.met_for is a STRICT floor (measured > threshold), so
# ">= 7" is expressed as "> 6.5".
PAIRED_SEEDS_THRESHOLD = MIN_PAIRED_SEEDS - 0.5

N_TESTS = 4                     # 2 regimes x 2 trained arms
ALPHA_CORRECTED = 0.05 / N_TESTS

NUM_SLOTS = 16
LATENT_DIM = 64

WRITE_CALLS_FLOOR = 200.0       # 956/943/970's P0 floor (Regime A)

# --- The redesigned instrument (see AUTHORING-TIME HEADROOM MEASUREMENT) ---
PROBE_CLUSTERS = 8
PROBE_JITTER = 0.078
PROBE_N = 1500
CLUSTER_BASE_SCALE = 0.078      # 956's own base scale
MI_N_SHUFFLE = 200              # label-shuffle null draws per probe
NMI_MARGIN = 0.10              # pre-registered elevation bar on NMI_excess
NMI_DV_BOUNDS = (0.0, 1.0)      # NMI_raw support; NMI_excess can dip below 0
                                # by estimator noise (ceiling is what matters)
N_STEPS_B = 15000
PRINT_EVERY_B = 2000
TAGGER_MOVE_EPS = 1e-8


# ------------------------------------------------------------------ #
# Objectives                                                                 #
# ------------------------------------------------------------------ #

def h1_contrastive_loss_k(write_addr_tagger, class_batches: Sequence[torch.Tensor],
                          tau: float = 1.0) -> torch.Tensor:
    """K-class generalisation of 970's `h1_context_divergence_loss`: the
    mean over all class PAIRS of -L1(mean selection distribution of class i,
    mean selection distribution of class j), selection distributions being
    F.softmax(-scores / tau) per example (the substrate's "lower score wins"
    convention, as compute_write_addressing_loss). For K = 2 this is exactly
    970's loss. Range [-2, 0]; minimised.
    """
    means = [F.softmax(-write_addr_tagger(b) / tau, dim=-1).mean(dim=0)
             for b in class_batches]
    total = torch.zeros((), dtype=means[0].dtype, device=means[0].device)
    n_pairs = 0
    for i, j in itertools.combinations(range(len(means)), 2):
        total = total + (means[i] - means[j]).abs().sum()
        n_pairs += 1
    return -total / max(n_pairs, 1)


# ------------------------------------------------------------------ #
# The MI instrument                                                          #
# ------------------------------------------------------------------ #

def _contingency(labels: Sequence[int], slots: Sequence[int], k: int) -> torch.Tensor:
    table = torch.zeros(k, NUM_SLOTS, dtype=torch.long)
    for c, s in zip(labels, slots):
        table[int(c), int(s)] += 1
    return table


def _mi_bits(table: torch.Tensor) -> Tuple[float, float, float]:
    """Plug-in I(class; slot), H(class), H(slot) in bits from a count table."""
    t = table.to(torch.float64)
    n = float(t.sum())
    if n <= 0:
        return 0.0, 0.0, 0.0
    p_c = t.sum(1) / n
    p_s = t.sum(0) / n
    p_cs = t / n
    nz = p_cs > 0
    outer = p_c.unsqueeze(1) * p_s.unsqueeze(0)
    mi = float((p_cs[nz] * torch.log2(p_cs[nz] / outer[nz])).sum())
    h_c = float(-(p_c[p_c > 0] * torch.log2(p_c[p_c > 0])).sum())
    h_s = float(-(p_s[p_s > 0] * torch.log2(p_s[p_s > 0])).sum())
    return mi, h_c, h_s


def nmi_excess_readout(labels: Sequence[int], slots: Sequence[int], k: int,
                       n_shuffle: int, shuffle_seed: int) -> Dict[str, Any]:
    """The instrument. NMI_raw = I/H(class); NMI_excess = NMI_raw minus the
    mean NMI of `n_shuffle` label permutations over the SAME slot sequence
    (finite-sample bias removal). Returns every intermediate plus the
    contingency table, so the manifest carries the full per-draw record.
    """
    table = _contingency(labels, slots, k)
    mi, h_c, h_s = _mi_bits(table)
    nmi_raw = mi / h_c if h_c > 0 else float("nan")
    gen = torch.Generator().manual_seed(shuffle_seed)
    lab = torch.tensor([int(x) for x in labels], dtype=torch.long)
    nulls: List[float] = []
    for _ in range(n_shuffle):
        perm = lab[torch.randperm(len(lab), generator=gen)].tolist()
        mi_p, h_cp, _ = _mi_bits(_contingency(perm, slots, k))
        nulls.append(mi_p / h_cp if h_cp > 0 else float("nan"))
    finite = [x for x in nulls if x == x]
    null_mean = sum(finite) / len(finite) if finite else float("nan")
    null_sd = (
        (sum((x - null_mean) ** 2 for x in finite) / max(len(finite) - 1, 1)) ** 0.5
        if len(finite) > 1 else float("nan")
    )
    spread_frac = float(((table > 0).sum(1) >= 2).to(torch.float32).mean()) if k > 0 else 0.0
    return {
        "n_draws": int(table.sum()),
        "k_classes": k,
        "mi_bits": mi,
        "h_class_bits": h_c,
        "h_slot_bits": h_s,
        "nmi_raw": nmi_raw,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "n_shuffle": n_shuffle,
        "nmi_excess": (nmi_raw - null_mean) if (nmi_raw == nmi_raw and null_mean == null_mean) else float("nan"),
        "class_spread_frac": spread_frac,  # fraction of classes routed to >= 2 slots
        "n_occupied_slots": int((table.sum(0) > 0).sum()),
        "contingency": table.tolist(),
    }


# ------------------------------------------------------------------ #
# Finding F4/F5 -- paired sign-flip permutation test (970 verbatim).        #
# ------------------------------------------------------------------ #

def _permutation_test_pvalue(diffs: List[float], n_perm: int = 100_000,
                              perm_seed: int = 0) -> float:
    """One-sided exact paired sign-flip test for n <= 20 (every 2**n sign
    pattern enumerated), Monte-Carlo beyond. `diffs` = trained - untrained
    per matched seed; large positive mean = evidence the DV rose. Ties
    counted as extreme (conservative). Attainable floor 1/2**n_nonzero.
    """
    diffs_t = [float(d) for d in diffs]
    n = len(diffs_t)
    if n == 0:
        return float("nan")
    observed = sum(diffs_t) / n
    if n <= 20:
        extreme = 0
        total = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            permuted = sum(s * d for s, d in zip(signs, diffs_t)) / n
            if permuted >= observed - 1e-12:
                extreme += 1
            total += 1
        return extreme / total
    rng = torch.Generator().manual_seed(perm_seed)
    extreme = 0
    for _ in range(n_perm):
        signs = (torch.randint(0, 2, (n,), generator=rng) * 2 - 1).tolist()
        permuted = sum(s * d for s, d in zip(signs, diffs_t)) / n
        if permuted >= observed - 1e-12:
            extreme += 1
    return extreme / n_perm


def _paired_diffs(rows_baseline: List[Dict[str, Any]], rows_arm: List[Dict[str, Any]],
                  value_key: str) -> Tuple[List[float], List[int]]:
    arm_by_seed = {r["seed"]: r.get(value_key) for r in rows_arm}
    diffs: List[float] = []
    seeds: List[int] = []
    for r in rows_baseline:
        u = r.get(value_key)
        t = arm_by_seed.get(r["seed"])
        if u is None or t is None or u != u or t != t:
            continue
        diffs.append(t - u)
        seeds.append(r["seed"])
    return diffs, seeds


def _mean(vals: Sequence[float]) -> float:
    xs = [float(v) for v in vals if v is not None and v == v]
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------ #
# Env / agent helpers (956/970 verbatim env params)                          #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=10, num_hazards=1, num_resources=4, hazard_harm=0.02,
        env_drift_interval=50, env_drift_prob=0.05, proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18, proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=10, num_hazards=8, num_resources=4, hazard_harm=0.05,
        env_drift_interval=50, env_drift_prob=0.05, proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18, proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2, write_selection: str, waddr_w: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=ALPHA_WORLD,
        alpha_self=0.3,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        sd016_writepath_mode=SD016_WRITEPATH_MODE,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_sd039_anchor_payload=True,
        contextmemory_gated_content_write=CONTEXTMEMORY_GATED_CONTENT_WRITE,
        contextmemory_write_selection=write_selection,
        contextmemory_write_refractory_k=REFRACTORY_K,
        contextmemory_write_addressing_loss_weight=waddr_w,
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, "use_noise_floor=True did not construct agent.noise_floor"
    cm = agent.e1.context_memory
    assert cm.num_slots == NUM_SLOTS, f"ContextMemory num_slots {cm.num_slots} != {NUM_SLOTS}"
    assert cm.write_selection == write_selection, (
        f"config wiring regression: requested {write_selection!r}, got {cm.write_selection!r}")
    total_latent_dim = agent.e1.config.self_dim + agent.e1.config.world_dim
    assert total_latent_dim == LATENT_DIM, f"self_dim+world_dim={total_latent_dim} != {LATENT_DIM}"
    if write_selection == "gumbel_learned":
        assert cm.write_addr_tagger is not None, "gumbel_learned did not construct write_addr_tagger"
    else:
        assert cm.write_addr_tagger is None, "write_addr_tagger constructed outside gumbel_learned"
        assert cm.write_refractory_k == REFRACTORY_K, "refractory k wiring regression"
    assert float(getattr(agent.config, "contextmemory_write_addressing_loss_weight", 0.0)) == waddr_w
    cm.memory.requires_grad_(False)  # 436e/f Adam-drift neutralisation
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False)
    return BASELINE_TEMPERATURE


def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor, num_actions: int) -> int:
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            harms.append(agent.e3.harm_eval(zw_next).mean().item())
        eff_t = _effective_temperature(agent)
        probs = F.softmax(-torch.tensor(harms, dtype=torch.float32) / eff_t, dim=0)
        return int(torch.multinomial(probs, 1).item())


class _WriteSequenceTracker:
    def __init__(self) -> None:
        self.sequence: List[int] = []
        self._last_total = 0

    def poll(self, context_memory) -> None:
        total = int(context_memory.slot_write_counts.sum().item())
        if total > self._last_total:
            n_new = total - self._last_total
            idx = context_memory.last_write_index
            if idx is not None:
                assert n_new == 1, f"expected one write() per sense(), got {n_new}"
                self.sequence.append(int(idx))
            self._last_total = total


def _tagger_state(agent: REEAgent) -> Optional[Dict[str, torch.Tensor]]:
    tagger = agent.e1.context_memory.write_addr_tagger
    if tagger is None:
        return None
    return {k: v.clone() for k, v in tagger.state_dict().items()}


def _tagger_moved(agent: REEAgent, init_state) -> Tuple[Optional[bool], float]:
    if init_state is None:
        return None, 0.0
    final = agent.e1.context_memory.write_addr_tagger.state_dict()
    max_abs = 0.0
    for k, v0 in init_state.items():
        max_abs = max(max_abs, (final[k] - v0).abs().max().item())
    return max_abs > TAGGER_MOVE_EPS, max_abs


# ------------------------------------------------------------------ #
# Probes                                                                     #
# ------------------------------------------------------------------ #

def _cluster_bases(seed: int, k: int) -> Tuple[List[torch.Tensor], torch.Generator]:
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, LATENT_DIM, generator=gen) * CLUSTER_BASE_SCALE for _ in range(k)]
    return bases, gen


def _sample_class_batches(bases: List[torch.Tensor], gen: torch.Generator,
                          batch_per_cluster: int, jitter: float) -> List[torch.Tensor]:
    return [b.expand(batch_per_cluster, -1)
            + torch.randn(batch_per_cluster, LATENT_DIM, generator=gen) * jitter
            for b in bases]


def probe_synthetic(agent: REEAgent, bases: List[torch.Tensor], jitter: float, n: int,
                    gen_seed: int, n_shuffle: int) -> Dict[str, Any]:
    """Eval-mode, interleaved-class synthetic probe: draw n states round-robin
    over the K bases with fresh jitter noise, write() each, record the slot,
    return the MI readout with its contingency table. In eval mode
    gumbel_learned is deterministic argmin over tagger scores; refractory is
    deterministic given write history (the interleaving is what makes the
    history neutral across classes).
    """
    agent.eval()
    cm = agent.e1.context_memory
    gen = torch.Generator().manual_seed(gen_seed)
    k = len(bases)
    labels: List[int] = []
    slots: List[int] = []
    with torch.no_grad():
        for i in range(n):
            cid = i % k
            state = bases[cid] + torch.randn(1, LATENT_DIM, generator=gen) * jitter
            cm.write(state)
            labels.append(cid)
            slots.append(int(cm.last_write_index))
    return nmi_excess_readout(labels, slots, k, n_shuffle, gen_seed + 1)


def probe_heldout_real(agent: REEAgent, heldout_safe: List[torch.Tensor],
                       heldout_dangerous: List[torch.Tensor], n_shuffle: int,
                       shuffle_seed: int) -> Dict[str, Any]:
    """Regime A's LOAD-BEARING probe: balanced to min(n_safe, n_dangerous)
    (the most recent states of the larger class are used), class-interleaved,
    eval mode, K = 2 real contexts. Returns the MI readout plus the balanced
    count; `n_used = 0` when either class is empty (the caller's heldout
    floor precondition scopes the arm out -- no number is fabricated).
    """
    n_used = min(len(heldout_safe), len(heldout_dangerous))
    if n_used == 0:
        out = nmi_excess_readout([], [], 2, 0, shuffle_seed)
        out["n_used_per_class"] = 0
        return out
    safe = heldout_safe[-n_used:]
    dang = heldout_dangerous[-n_used:]
    agent.eval()
    cm = agent.e1.context_memory
    labels: List[int] = []
    slots: List[int] = []
    with torch.no_grad():
        for s_state, d_state in zip(safe, dang):
            cm.write(s_state)
            labels.append(0)
            slots.append(int(cm.last_write_index))
            cm.write(d_state)
            labels.append(1)
            slots.append(int(cm.last_write_index))
    out = nmi_excess_readout(labels, slots, 2, n_shuffle, shuffle_seed)
    out["n_used_per_class"] = n_used
    # red-team F5 audit: mean L2 norm of the z_self (dims 0:32) and z_world
    # (dims 32:64) halves per class. A safe/dangerous ratio far from 1 on
    # z_self would mean the probe is separating reset transients, not context.
    half = LATENT_DIM // 2
    with torch.no_grad():
        out["class_latent_norms"] = {
            "safe_z_self": float(torch.cat(safe, 0)[:, :half].norm(dim=-1).mean()),
            "safe_z_world": float(torch.cat(safe, 0)[:, half:].norm(dim=-1).mean()),
            "dangerous_z_self": float(torch.cat(dang, 0)[:, :half].norm(dim=-1).mean()),
            "dangerous_z_world": float(torch.cat(dang, 0)[:, half:].norm(dim=-1).mean()),
        }
    return out


def _substrate_addressing_loss_on_window(agent: REEAgent) -> Optional[float]:
    """red-team F4: the DIVERSITY_956 arm's TRAINED objective is
    compute_write_addressing_loss over the `sequence` agent.compute_prediction_
    loss() builds -- a window of at most prediction_horizon+1 CONSECUTIVE
    [z_self, z_world] states sliced at a random start of the experience buffer
    (ree_core/agent.py, the E1 prediction-loss batch). Its convergence must be
    monitored on THAT form, not on a balanced cross-context batch (whose value
    and trend need not track it). This reproduces the construction under
    no_grad; the random start consumes the global RNG exactly as the training
    call does, in every gumbel arm alike (RNG matching).
    """
    cm = agent.e1.context_memory
    if cm.write_addr_tagger is None or len(agent._world_experience_buffer) < 2:
        return None
    buf_len = len(agent._world_experience_buffer)
    horizon = agent.e1.config.prediction_horizon
    max_start = max(1, buf_len - 1)
    start_idx = int(torch.randint(0, max_start, (1,)).item())
    end_idx = min(start_idx + horizon + 1, buf_len)
    if end_idx - start_idx < 2:
        return None
    with torch.no_grad():
        self_seq = agent._self_experience_buffer[start_idx:end_idx]
        world_seq = agent._world_experience_buffer[start_idx:end_idx]
        states = torch.stack([torch.cat([a.squeeze(0), w.squeeze(0)])
                              for a, w in zip(self_seq, world_seq)]).detach()
        return float(cm.compute_write_addressing_loss(states).item())


# ------------------------------------------------------------------ #
# Objective-convergence bookkeeping                                          #
# ------------------------------------------------------------------ #

def _convergence(trajectory: List[float], objective: Optional[str]) -> Dict[str, Any]:
    """start/end means over the first/last CONV_EDGE_FRAC of the monitored
    objective; `decrease_frac` = (start - end) / OBJECTIVE_RANGE. NaN when
    nothing was monitored (the precondition then reads unmet).
    """
    vals = [float(v) for v in trajectory if v == v]
    if objective is None or len(vals) < 2:
        return {"n_points": len(vals), "start_mean": float("nan"), "end_mean": float("nan"),
                "decrease_frac": float("nan")}
    edge = max(1, int(round(len(vals) * CONV_EDGE_FRAC)))
    start = sum(vals[:edge]) / edge
    end = sum(vals[-edge:]) / edge
    return {"n_points": len(vals), "start_mean": start, "end_mean": end,
            "decrease_frac": (start - end) / OBJECTIVE_RANGE[objective],
            "objective_range": OBJECTIVE_RANGE[objective]}


def _subsample(vals: List[float], max_points: int = 300) -> List[float]:
    if len(vals) <= max_points:
        return list(vals)
    stride = math.ceil(len(vals) / max_points)
    return list(vals[::stride])


# ------------------------------------------------------------------ #
# REGIME A -- real-agent episode/cell runners                               #
# ------------------------------------------------------------------ #

def _run_episode_a(agent: REEAgent, env: CausalGridWorldV2, steps: int, optimizer,
                   harm_eval_opt, harm_buf_pos, harm_buf_neg, states_safe, states_dangerous,
                   is_safe_ep: bool, tracker: _WriteSequenceTracker, arm: Dict[str, Any],
                   train: bool, trajectory: List[float], sampled_state_ids: Set[int],
                   trajectory_balanced: List[float], step_index_log: List[int]) -> None:
    """956's episode runner plus (a) write-stream state capture into the
    per-context buffer, (b) the H1 contrastive injection every
    H1_INJECT_EVERY_N_STEPS steps (H1_CONTRASTIVE arm), (c) objective
    monitoring at the same cadence (DIVERSITY_956: its own substrate loss
    evaluated under no_grad on a balanced recent batch). `train=False` is the
    held-out COLLECTION mode: agent in eval mode, no optimiser step of any
    kind, states appended to the buffers only.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    state_buf = states_safe if is_safe_ep else states_dangerous
    lambda_h1 = float(arm["lambda_h1"]) if train else 0.0

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)
        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

        # red-team F5: keep only step-matched states (within-episode step
        # index < STEP_MATCH_MAX) in BOTH classes, for training pools and
        # held-out alike. `_step` is 0-based, so this keeps steps 1..4.
        if _step < STEP_MATCH_MAX:
            obs_state = torch.cat([latent.z_self.detach(), latent.z_world.detach()], dim=-1).cpu()
            state_buf.append(obs_state)
            step_index_log.append(_step + 1)
            if len(state_buf) > MAX_STATE_BUF:
                del state_buf[:-MAX_STATE_BUF]

        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            agent._e1_tick(latent)
        tracker.poll(agent.e1.context_memory)

        z_world = latent.z_world.detach().clone()
        action_idx = _select_action_baseline(agent, z_world, env.action_dim)
        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh
        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        if train:
            e1_loss = agent.compute_prediction_loss()  # + substrate addressing loss when waddr_w > 0
            e2_loss = agent.compute_e2_loss()
            total = e1_loss + e2_loss

            cadence_hit = (_step + 1) % H1_INJECT_EVERY_N_STEPS == 0
            buffers_ready = len(states_safe) >= MIN_H1_BUF and len(states_dangerous) >= MIN_H1_BUF
            if cadence_hit and buffers_ready:
                # The balanced batch is drawn in EVERY gumbel_learned arm, including
                # UNTRAINED (which then discards it), so all three consume the
                # global RNG identically and their env/agent trajectories are
                # matched up to the tagger gradient itself -- the only thing that
                # differs between them. (REFRACTORY_K2 consumes no Gumbel noise in
                # write() and is a reference arm, not a matched one.)
                k = min(H1_BATCH_PER_SIDE, len(states_safe), len(states_dangerous))
                safe_idx = torch.randperm(len(states_safe))[:k].tolist()
                dang_idx = torch.randperm(len(states_dangerous))[:k].tolist()
                sampled_state_ids.update(id(states_safe[i]) for i in safe_idx)
                sampled_state_ids.update(id(states_dangerous[i]) for i in dang_idx)
                batch_safe = torch.cat([states_safe[i] for i in safe_idx], dim=0)
                batch_dang = torch.cat([states_dangerous[i] for i in dang_idx], dim=0)
                tagger = agent.e1.context_memory.write_addr_tagger
                if tagger is None:
                    pass  # REFRACTORY_K2: nothing to train or monitor
                elif lambda_h1 > 0.0:
                    h1_loss = h1_contrastive_loss_k(tagger, [batch_safe, batch_dang])
                    total = total + lambda_h1 * h1_loss
                    trajectory.append(float(h1_loss.item()))
                elif arm["objective"] == "addressing_diversity":
                    # Convergence gate reads the TRAINED form (window of the
                    # experience buffer, red-team F4); the balanced cross-
                    # context value is kept as a descriptive secondary.
                    mon_window = _substrate_addressing_loss_on_window(agent)
                    if mon_window is not None:
                        trajectory.append(mon_window)
                    with torch.no_grad():
                        mon_bal = agent.e1.context_memory.compute_write_addressing_loss(
                            torch.cat([batch_safe, batch_dang], dim=0))
                    trajectory_balanced.append(float(mon_bal.item()))

            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if is_harm:
                harm_buf_pos.append(z_world)
            else:
                harm_buf_neg.append(z_world)
            if len(harm_buf_pos) > MAX_HARM_BUF:
                del harm_buf_pos[:-MAX_HARM_BUF]
            if len(harm_buf_neg) > MAX_HARM_BUF:
                del harm_buf_neg[:-MAX_HARM_BUF]
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_b = torch.cat([torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0),
                                  torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)], dim=0)
                target_t = torch.cat([torch.ones(k_pos, 1, device=agent.device),
                                      torch.zeros(k_neg, 1, device=agent.device)], dim=0)
                pred = agent.e3.harm_eval_head(zw_b)
                h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
                harm_eval_opt.zero_grad()
                h_loss.backward()
                harm_eval_opt.step()

        if done:
            break


def _run_cell_a(arm: Dict[str, Any], seed: int, base_config_slice: Dict[str, Any],
                zg: ZGoalStreamAccumulator, training_episodes: int, heldout_episodes: int,
                steps_per_episode: int, context_switch_every: int, probe_n: int,
                n_shuffle: int) -> Dict[str, Any]:
    arm_name = arm["name"]
    cell_config_slice = {**base_config_slice, "regime": "A", "arm": arm_name,
                         "write_selection": arm["write_selection"],
                         "lambda_h1": arm["lambda_h1"], "waddr_w": arm["waddr_w"]}
    # UNTRAINED is the reusable baseline cell: minted cross-driver (mint-as-you-go).
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=(arm_name != BASELINE_ARM)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe, arm["write_selection"], arm["waddr_w"])
        tagger_init = _tagger_state(agent)

        standard_params = [p for n, p in agent.named_parameters()
                           if "harm_eval_head" not in n and "context_memory.memory" not in n]
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        states_safe: List[torch.Tensor] = []
        states_dangerous: List[torch.Tensor] = []
        heldout_safe: List[torch.Tensor] = []
        heldout_dangerous: List[torch.Tensor] = []
        tracker = _WriteSequenceTracker()
        sampled_state_ids: Set[int] = set()
        trajectory: List[float] = []
        trajectory_balanced: List[float] = []
        step_log_train_safe: List[int] = []
        step_log_train_dang: List[int] = []
        step_log_held_safe: List[int] = []
        step_log_held_dang: List[int] = []
        n_train_writes = 0

        total_episodes = training_episodes + heldout_episodes
        print(f"Seed {seed} Condition A_{arm_name}", flush=True)
        agent.train()
        for ep in range(total_episodes):
            block = ep // context_switch_every
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            train = ep < training_episodes
            if not train:
                if ep == training_episodes:
                    n_train_writes = len(tracker.sequence)  # red-team N3: training-phase count only
                agent.eval()
                _run_episode_a(agent, env, steps_per_episode, None, None, harm_buf_pos,
                               harm_buf_neg, heldout_safe, heldout_dangerous, is_safe_ep,
                               tracker, arm, False, trajectory, sampled_state_ids,
                               trajectory_balanced,
                               step_log_held_safe if is_safe_ep else step_log_held_dang)
            else:
                _run_episode_a(agent, env, steps_per_episode, optimizer, harm_eval_opt,
                               harm_buf_pos, harm_buf_neg, states_safe, states_dangerous,
                               is_safe_ep, tracker, arm, True, trajectory, sampled_state_ids,
                               trajectory_balanced,
                               step_log_train_safe if is_safe_ep else step_log_train_dang)
            if (ep + 1) % 20 == 0 or (ep + 1) == total_episodes:
                phase = "train" if train else "heldout-collect"
                print(f"  [train] regime=A arm={arm_name} seed={seed} ep {ep + 1}/{total_episodes} "
                      f"phase={phase} n_writes={len(tracker.sequence)} "
                      f"n_states_safe={len(states_safe)} n_states_dangerous={len(states_dangerous)} "
                      f"n_heldout_safe={len(heldout_safe)} n_heldout_dangerous={len(heldout_dangerous)} "
                      f"last_objective={trajectory[-1] if trajectory else None}", flush=True)

        if heldout_episodes == 0:
            n_train_writes = len(tracker.sequence)
        n_total_writes = len(tracker.sequence)
        zg.observe(agent)
        moved, max_abs = _tagger_moved(agent, tagger_init)

        # Held-out states were collected into separate lists after training,
        # so no sampler can ever have drawn them; asserted rather than assumed.
        heldout_ids = {id(s) for s in heldout_safe} | {id(s) for s in heldout_dangerous}
        assert not (heldout_ids & sampled_state_ids), "held-out state was drawn into a training batch"

        probe_real = probe_heldout_real(agent, heldout_safe, heldout_dangerous, n_shuffle,
                                        seed + 700_000)
        heldout_adequate = probe_real["n_used_per_class"] >= MIN_HELDOUT_PER_CLASS
        # An inadequate held-out split contributes NO load-bearing value for
        # this seed (dropped from the paired test), but the readout it did
        # produce is still recorded for audit.
        heldout_lb_value = probe_real["nmi_excess"] if heldout_adequate else None
        fresh_bases, _ = _cluster_bases(seed + 500_000, PROBE_CLUSTERS)
        probe_fresh = probe_synthetic(agent, fresh_bases, PROBE_JITTER, probe_n,
                                      seed + 800_000, n_shuffle)
        conv = _convergence(trajectory, arm["objective"] if arm["trained"] else None)

        row: Dict[str, Any] = {
            "regime": "A", "arm": arm_name, "seed": seed,
            "write_selection": arm["write_selection"],
            "lambda_h1": arm["lambda_h1"], "waddr_w": arm["waddr_w"],
            "n_write_calls_training": n_train_writes,
            "n_write_calls_total_incl_heldout": n_total_writes,
            "n_states_safe_raw": len(states_safe),
            "n_states_dangerous_raw": len(states_dangerous),
            "n_heldout_safe_raw": len(heldout_safe),
            "n_heldout_dangerous_raw": len(heldout_dangerous),
            "n_heldout_used_per_class": probe_real["n_used_per_class"],
            "heldout_adequate": heldout_adequate,
            "step_match_max": STEP_MATCH_MAX,
            "step_index_histogram": {
                "train_safe": [step_log_train_safe.count(i) for i in range(1, STEP_MATCH_MAX + 1)],
                "train_dangerous": [step_log_train_dang.count(i) for i in range(1, STEP_MATCH_MAX + 1)],
                "heldout_safe": [step_log_held_safe.count(i) for i in range(1, STEP_MATCH_MAX + 1)],
                "heldout_dangerous": [step_log_held_dang.count(i) for i in range(1, STEP_MATCH_MAX + 1)],
            },
            "tagger_params_moved": moved,
            "tagger_max_abs_param_diff": max_abs,
            # LOAD-BEARING (Regime A): held-out real 2-context MI readout;
            # None when the split was inadequate (dropped from the paired test)
            "heldout_real_nmi_excess": heldout_lb_value,
            "heldout_real_readout": probe_real,
            "objective_trajectory_balanced_batch": _subsample(trajectory_balanced),
            # SECONDARY: synthetic fresh K-cluster probe on the same tagger
            "synthetic_fresh_nmi_excess": probe_fresh["nmi_excess"],
            "synthetic_fresh_readout": probe_fresh,
            "objective": arm["objective"],
            "objective_trajectory": _subsample(trajectory),
            "objective_convergence": conv,
        }
        cell.stamp(row)

    print(f"verdict: {'PASS' if n_train_writes >= WRITE_CALLS_FLOOR else 'FAIL'} "
          f"(regime=A writepath engagement; n_write_calls={n_train_writes}, "
          f"heldout_nmi_excess={probe_real['nmi_excess']}, n_used={probe_real['n_used_per_class']}, "
          f"adequate={heldout_adequate}, "
          f"synthetic_fresh_nmi_excess={probe_fresh['nmi_excess']:.3f})", flush=True)
    return row


# ------------------------------------------------------------------ #
# REGIME B -- synthetic K-cluster SGD, no environment                        #
# ------------------------------------------------------------------ #

def _train_regime_b(agent: REEAgent, arm: Dict[str, Any], bases: List[torch.Tensor],
                    gen: torch.Generator, n_steps: int, seed: int, print_every: int,
                    trajectory: List[float]) -> None:
    if n_steps <= 0 or not arm["trained"]:
        return
    cm = agent.e1.context_memory
    optimizer = optim.Adam(cm.write_addr_tagger.parameters(), lr=1e-3)
    for step in range(n_steps):
        batches = _sample_class_batches(bases, gen, H1_BATCH_PER_SIDE, PROBE_JITTER)
        if arm["objective"] == "contrastive":
            loss = h1_contrastive_loss_k(cm.write_addr_tagger, batches)
            scaled = arm["lambda_h1"] * loss
        else:
            loss = cm.compute_write_addressing_loss(torch.cat(batches, dim=0))
            scaled = arm["waddr_w"] * loss
        optimizer.zero_grad()
        scaled.backward()
        optimizer.step()
        trajectory.append(float(loss.item()))
        if (step + 1) % print_every == 0 or (step + 1) == n_steps:
            print(f"  [train] regime=B arm={arm['name']} seed={seed} step {step + 1}/{n_steps} "
                  f"objective={trajectory[-1]:.4f}", flush=True)


def _run_cell_b(arm: Dict[str, Any], seed: int, base_config_slice: Dict[str, Any],
                zg: ZGoalStreamAccumulator, n_steps: int, print_every: int, probe_n: int,
                n_shuffle: int) -> Dict[str, Any]:
    arm_name = arm["name"]
    cell_config_slice = {**base_config_slice, "regime": "B", "arm": arm_name,
                         "write_selection": arm["write_selection"],
                         "lambda_h1": arm["lambda_h1"], "waddr_w": arm["waddr_w"],
                         "n_steps": n_steps if arm["trained"] else 0}
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=(arm_name != BASELINE_ARM)) as cell:
        env_throwaway = _make_env_safe(seed)  # dims only; never reset()/step()ped
        agent = _make_agent(env_throwaway, arm["write_selection"], arm["waddr_w"])
        tagger_init = _tagger_state(agent)
        bases, gen = _cluster_bases(seed, PROBE_CLUSTERS)
        trajectory: List[float] = []
        print(f"Seed {seed} Condition B_{arm_name}", flush=True)
        agent.train()
        _train_regime_b(agent, arm, bases, gen, n_steps, seed, print_every, trajectory)
        zg.observe(agent)
        moved, max_abs = _tagger_moved(agent, tagger_init)

        # SECONDARY: the clusters the arm trained on, fresh noise (970's F1 readout).
        probe_trained = probe_synthetic(agent, bases, PROBE_JITTER, probe_n, seed + 800_000, n_shuffle)
        # LOAD-BEARING: K fresh, disjoint cluster bases.
        fresh_bases, _ = _cluster_bases(seed + 500_000, PROBE_CLUSTERS)
        probe_fresh = probe_synthetic(agent, fresh_bases, PROBE_JITTER, probe_n, seed + 900_000, n_shuffle)
        conv = _convergence(trajectory, arm["objective"] if arm["trained"] else None)

        row: Dict[str, Any] = {
            "regime": "B", "arm": arm_name, "seed": seed,
            "write_selection": arm["write_selection"],
            "lambda_h1": arm["lambda_h1"], "waddr_w": arm["waddr_w"],
            "n_steps": n_steps if arm["trained"] else 0,
            "tagger_params_moved": moved,
            "tagger_max_abs_param_diff": max_abs,
            "synthetic_fresh_nmi_excess": probe_fresh["nmi_excess"],
            "synthetic_fresh_readout": probe_fresh,
            "synthetic_trained_nmi_excess": probe_trained["nmi_excess"],
            "synthetic_trained_readout": probe_trained,
            "objective": arm["objective"],
            "objective_trajectory": _subsample(trajectory),
            "objective_convergence": conv,
        }
        cell.stamp(row)

    engaged = (not arm["trained"]) or bool(moved)
    print(f"verdict: {'PASS' if engaged else 'FAIL'} (regime=B training-engagement control; "
          f"fresh_nmi_excess={probe_fresh['nmi_excess']:.3f}, "
          f"trained_nmi_excess={probe_trained['nmi_excess']:.3f})", flush=True)
    return row


# ------------------------------------------------------------------ #
# Top-level run                                                              #
# ------------------------------------------------------------------ #

def _build_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="writepath_engaged",
            description="min n_write_calls (training phase) over the regime's UNTRAINED and this arm's cells",
            control="ContextMemory.write() genuinely fired during real training (956/943/970 floor)",
            threshold=WRITE_CALLS_FLOOR, direction="lower",
            applies_to=lambda ctx: ctx["regime"] == "A",
            applies_note="Regime B has no environment and no write() during training; its "
                         "engagement is by construction (N_STEPS_B fixed) and is separately "
                         "controlled by tagger_params_moved.",
        ),
        PreconditionSpec(
            name="heldout_adequate_pairs",
            description=f"number of seeds whose UNTRAINED and arm cells BOTH realised >= "
                        f"{MIN_HELDOUT_PER_CLASS} balanced held-out states per class "
                        f"(>= {MIN_PAIRED_SEEDS}; strict floor at {PAIRED_SEEDS_THRESHOLD})",
            control="held-out split sized as a fraction of realised episodes, collected with every "
                    "optimiser stopped; an inadequate cell is dropped from the paired test rather "
                    "than fabricating a value, and the unit is scored only with enough pairs for "
                    "the sign-flip p-floor to clear alpha",
            threshold=PAIRED_SEEDS_THRESHOLD, direction="lower",
            applies_to=lambda ctx: ctx["regime"] == "A",
            applies_note="Regime B's fresh-cluster probe has a fixed PROBE_N draws by construction.",
        ),
        PreconditionSpec(
            name="dv_headroom_nmi_excess",
            description="ceiling headroom of the load-bearing DV above the UNTRAINED control arm: "
                        "1.0 - max_seed(NMI_excess(UNTRAINED)) must exceed NMI_MARGIN "
                        "(experiments/_metrics.dv_headroom_check, statistic ceiling_headroom)",
            control="the seed-matched UNTRAINED arm's realised NMI_excess on the SAME load-bearing readout",
            threshold=NMI_MARGIN, direction="lower", kind="dv_headroom",
        ),
        PreconditionSpec(
            name="objective_converged",
            description="worst-cell (start_mean - end_mean) / objective_range over this arm's seeds "
                        f"must exceed {CONV_MIN_DECREASE_FRAC} (first/last {CONV_EDGE_FRAC:.0%} of the trajectory)",
            control="the arm's OWN monitored objective (971's coupling loss rose across training)",
            threshold=CONV_MIN_DECREASE_FRAC, direction="lower",
        ),
    ]


def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()
    # Dry-run schedule is sized so BOTH per-context buffers reach MIN_H1_BUF and
    # the H1 injection / DIVERSITY monitor actually fire (dangerous episodes end
    # in ~4 steps, so 5 dangerous episodes are needed to bank >= 8 states) --
    # a smoke that never reaches the manipulated code path is blind.
    training_episodes = 10 if dry_run else TRAINING_EPISODES
    heldout_episodes = 4 if dry_run else HELDOUT_EPISODES
    steps_per_episode = 15 if dry_run else STEPS_PER_EPISODE
    context_switch_every = 1 if dry_run else CONTEXT_SWITCH_EVERY
    n_steps_b = 6 if dry_run else N_STEPS_B
    print_every_b = 2 if dry_run else PRINT_EVERY_B
    probe_n = 40 if dry_run else PROBE_N
    n_shuffle = 20 if dry_run else MI_N_SHUFFLE
    seeds = SEEDS[:1] if dry_run else SEEDS

    base_config_slice: Dict[str, Any] = {
        "seeds": seeds, "lambda_h1": LAMBDA_H1,
        "write_addressing_loss_weight_956": WRITE_ADDRESSING_LOSS_WEIGHT_956,
        "refractory_k": REFRACTORY_K,
        "training_episodes": training_episodes, "heldout_episodes": heldout_episodes,
        "steps_per_episode": steps_per_episode, "context_switch_every": context_switch_every,
        "n_steps_b": n_steps_b,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE, "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR, "num_slots": NUM_SLOTS, "latent_dim": LATENT_DIM,
        "probe_n": probe_n, "probe_jitter": PROBE_JITTER, "probe_clusters": PROBE_CLUSTERS,
        "cluster_base_scale": CLUSTER_BASE_SCALE, "mi_n_shuffle": n_shuffle,
    }

    specs = _build_specs()
    units = [{"id": f"{regime}::{arm}", "regime": regime, "arm": arm}
             for regime in ("A", "B") for arm in TRAINED_ARMS]
    gate_audit = assert_no_structurally_unsatisfiable_gate(specs, units)

    rows_a: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows_a.append(_run_cell_a(arm, seed, base_config_slice, zg, training_episodes,
                                      heldout_episodes, steps_per_episode, context_switch_every,
                                      probe_n, n_shuffle))
    rows_b: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows_b.append(_run_cell_b(arm, seed, base_config_slice, zg, n_steps_b,
                                      print_every_b, probe_n, n_shuffle))

    by_arm = {"A": {a["name"]: [r for r in rows_a if r["arm"] == a["name"]] for a in ARMS},
              "B": {a["name"]: [r for r in rows_b if r["arm"] == a["name"]] for a in ARMS}}
    LB_KEY = {"A": "heldout_real_nmi_excess", "B": "synthetic_fresh_nmi_excess"}

    # --- dv_headroom entries (one per regime, from the UNTRAINED control) ---
    headroom_entries: Dict[str, Dict[str, Any]] = {}
    for regime in ("A", "B"):
        control_vals = [r[LB_KEY[regime]] for r in by_arm[regime][BASELINE_ARM]
                        if r[LB_KEY[regime]] is not None and r[LB_KEY[regime]] == r[LB_KEY[regime]]]
        if control_vals:
            headroom_entries[regime] = dv_headroom_check(
                f"dv_headroom_nmi_excess_regime_{regime}",
                dv_name=f"NMI_excess ({LB_KEY[regime]})",
                criterion_threshold=NMI_MARGIN, control_values=control_vals,
                statistic="ceiling_headroom", dv_bounds=NMI_DV_BOUNDS, margin=1.0,
                control_arm=BASELINE_ARM, regime=regime, control_values_per_seed=control_vals)
        else:
            headroom_entries[regime] = {"name": f"dv_headroom_nmi_excess_regime_{regime}",
                                        "kind": "dv_headroom", "measured": float("nan"),
                                        "threshold": NMI_MARGIN, "direction": "lower",
                                        "dv_name": f"NMI_excess ({LB_KEY[regime]})",
                                        "achievable_statistic": "ceiling_headroom",
                                        "note": "no finite control values -- UNTRAINED arm produced no readout"}

    # --- per-(regime, trained arm) gates ---
    gates: Dict[str, Dict[str, Any]] = {}
    convergence_by_unit: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        regime, arm = unit["regime"], unit["arm"]
        arm_rows = by_arm[regime][arm]
        ctrl_rows = by_arm[regime][BASELINE_ARM]
        convs = [(r["objective_convergence"]["decrease_frac"], r["seed"]) for r in arm_rows]
        finite = [(v, s) for v, s in convs if v == v]
        worst = min(finite, key=lambda x: x[0]) if finite else (float("nan"), None)
        convergence_by_unit[unit["id"]] = {"worst_decrease_frac": worst[0], "offending_cell_seed": worst[1],
                                           "per_seed": [{"seed": s, "decrease_frac": v} for v, s in convs]}
        measured = {"dv_headroom_nmi_excess": float(headroom_entries[regime]["measured"]),
                    "objective_converged": float(worst[0])}
        if regime == "A":
            measured["writepath_engaged"] = float(min(
                [r["n_write_calls_training"] for r in arm_rows + ctrl_rows], default=0))
            ctrl_ok = {r["seed"] for r in ctrl_rows if r["heldout_adequate"]}
            arm_ok = {r["seed"] for r in arm_rows if r["heldout_adequate"]}
            measured["heldout_adequate_pairs"] = float(len(ctrl_ok & arm_ok))
        gates[unit["id"]] = evaluate_arm_gate(unit["id"], unit, specs, measured=measured)
    gate = aggregate_arm_gates([gates[u["id"]] for u in units])

    # --- statistics per (regime, trained arm) ---
    tests: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        regime, arm = unit["regime"], unit["arm"]
        diffs, dseeds = _paired_diffs(by_arm[regime][BASELINE_ARM], by_arm[regime][arm], LB_KEY[regime])
        p = _permutation_test_pvalue(diffs) if diffs else float("nan")
        mean_d = _mean(diffs) if diffs else float("nan")
        n_ties = sum(1 for d in diffs if abs(d) < 1e-12)
        green = gates[unit["id"]]["gate_green"]
        passed = bool(green and mean_d == mean_d and mean_d >= NMI_MARGIN and p == p and p < ALPHA_CORRECTED)
        tests[unit["id"]] = {
            "regime": regime, "arm": arm, "readout": LB_KEY[regime],
            "mean_untrained": _mean([r[LB_KEY[regime]] for r in by_arm[regime][BASELINE_ARM]]),
            "mean_arm": _mean([r[LB_KEY[regime]] for r in by_arm[regime][arm]]),
            "paired_diffs": diffs, "paired_seeds": dseeds, "n_paired_seeds": len(diffs),
            "n_tied_pairs": n_ties, "mean_diff": mean_d, "required_margin": NMI_MARGIN,
            "p_value": p, "alpha_corrected": ALPHA_CORRECTED,
            "attainable_p_floor": (0.5 ** max(len(diffs) - n_ties, 0)) if diffs else float("nan"),
            "gate_green": green, "passed": passed,
            "non_degenerate": bool(green and len(diffs) >= 2 and (max(diffs) - min(diffs)) > 1e-9),
        }

    pass_h1 = {r: tests[f"{r}::H1_CONTRASTIVE"]["passed"] for r in ("A", "B")}
    pass_div = {r: tests[f"{r}::DIVERSITY_956"]["passed"] for r in ("A", "B")}
    green_h1 = {r: gates[f"{r}::H1_CONTRASTIVE"]["gate_green"] for r in ("A", "B")}
    green_div = {r: gates[f"{r}::DIVERSITY_956"]["gate_green"] for r in ("A", "B")}
    h1_regimes = [r for r in ("A", "B") if pass_h1[r]]
    overall_pass = bool(h1_regimes)
    # red-team F2: the run is instrument-not-ready when NEITHER H1 unit is
    # scorable, whatever the DIVERSITY units did -- H1 is the load-bearing arm.
    h1_any_green = any(green_h1.values())
    # red-team F1: the necessity sub-claim is evaluable only where the
    # content-blind control was itself SCORED (gate green); a gate-red control
    # is unscored, never a refutation.
    required_evaluable = bool(h1_regimes) and all(green_div[r] for r in h1_regimes)
    required_holds = (all(not pass_div[r] for r in h1_regimes) if required_evaluable else None)
    pattern = "_".join(f"{r.lower()}_h1{'pass' if pass_h1[r] else 'fail'}"
                       f"_div{'pass' if pass_div[r] else ('fail' if green_div[r] else 'unscored')}"
                       for r in ("A", "B"))
    suffix = {"AB": "both_regimes", "A": "real_agent_only", "B": "synthetic_only"}.get("".join(h1_regimes), "")

    if not h1_any_green:
        label = "instrument_gate_not_ready"
    elif overall_pass and required_holds is None:
        label = f"h1_confirmed_content_reference_necessity_unscored_{suffix}"
    elif overall_pass and required_holds:
        label = f"h1_confirmed_content_reference_required_{suffix}"
    elif overall_pass:
        # red-team N1: carry the per-regime pattern in the label itself.
        label = f"training_raises_content_conditioning_content_reference_not_required_{pattern}"
    elif any(pass_div.values()):
        label = "h1_inverted_content_blind_objective_suffices"
    else:
        label = "h1_not_confirmed_on_mi_instrument"
    h4_caveat = ("synthetic-regime-only positive: points at the input distribution (H4 / SD-070); "
                 "residual ~10x update-budget asymmetry between regimes (unchanged from 970) -- "
                 "suggestive, not decisive, for H4 attribution"
                 if h1_regimes == ["B"] else None)

    status = "PASS" if overall_pass else "FAIL"

    # Adjudication list: green units' preconditions when H1 is scorable
    # somewhere (the helper's partial-run rule); otherwise the H1 units' own
    # preconditions so the indexer sees WHICH gate vacated the load-bearing arm.
    if h1_any_green:
        adjudication_preconditions = gate["adjudication_preconditions"]
        degeneracy_reason = gate["degeneracy_reason"]
    else:
        adjudication_preconditions = [p for r in ("A", "B")
                                      for p in gates[f"{r}::H1_CONTRASTIVE"]["preconditions"]]
        degeneracy_reason = ("H1_CONTRASTIVE unit RED in BOTH regimes (" + "; ".join(
            f"{r}: {', '.join(gates[f'{r}::H1_CONTRASTIVE']['failed_preconditions'])}"
            for r in ("A", "B")) + ") -- the load-bearing arm was not scored; "
            + gate["degeneracy_reason"])

    criteria_non_degenerate = {
        "H1_contrastive_raises_generalising_content_conditioning_in_either_regime": h1_any_green,
        **{f"{u['id']}::directional_test": tests[u["id"]]["non_degenerate"] for u in units},
    }
    reference = {r: {"mean_refractory_k2": _mean([x[LB_KEY[r]] for x in by_arm[r][REFERENCE_ARM]]),
                     "mean_untrained": tests[f"{r}::H1_CONTRASTIVE"]["mean_untrained"],
                     "per_seed_refractory_k2": [x[LB_KEY[r]] for x in by_arm[r][REFERENCE_ARM]]}
                 for r in ("A", "B")}

    criteria = [
        {"name": "H1_contrastive_raises_generalising_content_conditioning_in_either_regime",
         "load_bearing": True, "passed": overall_pass,
         "regime_a_pass": pass_h1["A"], "regime_b_pass": pass_h1["B"],
         "combination_rule": ("OR across regimes: pass(A,H1) OR pass(B,H1); each regime's pass "
                              "requires its (regime,H1) gate green AND mean paired NMI_excess "
                              f"elevation >= {NMI_MARGIN} AND exact sign-flip p < {ALPHA_CORRECTED} "
                              "(Bonferroni over 2 regimes x 2 trained arms)"),
         "note": ("Load-bearing readouts: Regime A = held-out real 2-context NMI_excess; Regime B = "
                  "FRESH-cluster NMI_excess. A FAIL in one regime with a PASS in the other is not "
                  "'mostly failed' -- see combination_rule.") + (f" {h4_caveat}" if h4_caveat else "")},
        {"name": "H1_content_reference_required", "load_bearing": False, "kind": "secondary",
         "passed": bool(required_holds) if required_holds is not None else False,
         "evaluable": required_evaluable,
         "diversity_gate_green": green_div, "h1_gate_green": green_h1,
         "regime_pattern": pattern,
         "regime_a_diversity_pass": pass_div["A"], "regime_b_diversity_pass": pass_div["B"],
         "combination_rule": ("in every regime where H1 passes, DIVERSITY_956 was SCORED (gate green) "
                              "and does NOT pass; not evaluable when H1 passes nowhere or when the "
                              "control's gate is red in an H1-passing regime (unscored != refuted)"),
         "note": "H1's necessity sub-claim: content-BLIND training should not raise the DV."},
        {"name": "objective_convergence_per_unit", "load_bearing": False, "kind": "readiness",
         "per_unit": convergence_by_unit},
        {"name": "refractory_k2_reference_descriptive", "load_bearing": False, "kind": "descriptive",
         "per_regime": reference,
         "note": "interim-enablement reference (decision_2026_09_06); no test"},
    ]

    metrics: Dict[str, Any] = {
        "load_bearing_readout_key": LB_KEY,
        "per_seed_nmi_excess": {
            r: {a["name"]: [x[LB_KEY[r]] for x in by_arm[r][a["name"]]] for a in ARMS} for r in ("A", "B")},
        "per_seed_secondary_nmi_excess": {
            "A_synthetic_fresh": {a["name"]: [x["synthetic_fresh_nmi_excess"] for x in by_arm["A"][a["name"]]] for a in ARMS},
            "B_synthetic_trained": {a["name"]: [x["synthetic_trained_nmi_excess"] for x in by_arm["B"][a["name"]]] for a in ARMS},
        },
        "per_seed_heldout_used_per_class_regime_a": {
            a["name"]: [x["n_heldout_used_per_class"] for x in by_arm["A"][a["name"]]] for a in ARMS},
        "per_seed_n_write_calls_regime_a": {
            a["name"]: [x["n_write_calls_training"] for x in by_arm["A"][a["name"]]] for a in ARMS},
        "per_seed_class_spread_frac": {
            r: {a["name"]: [x[("heldout_real_readout" if r == "A" else "synthetic_fresh_readout")]["class_spread_frac"]
                             for x in by_arm[r][a["name"]]] for a in ARMS} for r in ("A", "B")},
        "tests": tests,
        "dv_headroom_entries": headroom_entries,
        "convergence_by_unit": convergence_by_unit,
        "gate_audit": gate_audit,
        # Flat scalar readout, MERGED INTO this `metrics` dict rather than emitted as a
        # sibling `readout` (REE_assembly evidence/planning/
        # flat_scalar_readout_recording_gap_20260909.md). The runpack converter takes the
        # FIRST non-empty of metrics / aggregates / summary_metrics / readout, so this
        # already-populated `metrics` would shadow a `readout` sibling entirely -- and
        # every entry above is keyed by regime, arm or unit, so it harvested zero NUMERIC
        # entries and the pack scored empty: no fail_if stop threshold could fire, the
        # duplicate-emission supersession fingerprint was skipped, and the index carried
        # no deltas. The nested per-unit blocks are kept unchanged beside these scalars.
        #
        # Three distinctions are recorded rather than collapsed, all of them red-team
        # findings this driver already encodes in its label grid: (F2) H1 is the
        # load-bearing arm, so h1_any_green is recorded separately -- a run where neither
        # H1 unit was scorable is instrument-not-ready, whatever DIVERSITY did; (F1) the
        # necessity sub-claim is EVALUABLE only where the content-blind control was
        # itself gate-green, so required_evaluable and required_holds are separate
        # scalars and a gate-red control records as unscored rather than as a refutation;
        # and each unit's gate_green is recorded beside its pass flag. flat_readout()
        # enforces the two encoding rules (bools -> 0/1 ints; non-finite/None dropped --
        # which matters here: mean_diff and p_value are genuinely NaN for a unit with no
        # paired seeds, and dropping them is correct). Recording-only: the verdict grid,
        # criteria, thresholds and DVs are unchanged.
        **flat_readout(dict(
            {
                "H1_raises_content_conditioning_in_either_regime": overall_pass,
                "n_h1_regimes_passing": len(h1_regimes),
                "n_regimes": 2,
                "h1_any_green": h1_any_green,
                "required_evaluable": required_evaluable,
                "required_holds": required_holds,
                "nmi_margin": NMI_MARGIN,
                "alpha_corrected": ALPHA_CORRECTED,
                "non_degenerate_flag": gate["non_degenerate"],
                "n_units": len(units),
                "n_units_green": sum(
                    1 for u in units if gates[u["id"]]["gate_green"]),
                "n_cells": len(rows_a) + len(rows_b),
            },
            # per-unit statistics: which unit carried (or vacated) the result
            **{
                f"{_u['id'].replace('::', '_')}_{_k}": tests[_u["id"]][_k]
                for _u in units
                for _k in ("mean_untrained", "mean_arm", "mean_diff", "p_value",
                           "n_paired_seeds", "n_tied_pairs", "attainable_p_floor",
                           "gate_green", "passed", "non_degenerate")
            },
        )),
    }

    lines = [f"# {QUEUE_ID} -- ContextMemory write-content H1 on the redesigned MI instrument (supersedes {SUPERSEDES})",
             "", f"**Status:** {status}  **Label:** {label}", "",
             "| unit | readout | mean UNTRAINED | mean arm | mean diff | p | gate | pass |",
             "|---|---|---|---|---|---|---|---|"]
    for u in units:
        t = tests[u["id"]]
        lines.append(f"| {u['id']} | {t['readout']} | {t['mean_untrained']:.3f} | {t['mean_arm']:.3f} | "
                     f"{t['mean_diff']:.3f} | {t['p_value']:.4f} | {t['gate_green']} | {t['passed']} |")
    lines += ["", f"refractory k=2 reference: A {reference['A']['mean_refractory_k2']:.3f}, "
                  f"B {reference['B']['mean_refractory_k2']:.3f}", "",
              f"margin {NMI_MARGIN}, alpha_corrected {ALPHA_CORRECTED}", degeneracy_reason or ""]
    if h4_caveat:
        lines.append(f"\nH4 caveat: {h4_caveat}")

    result: Dict[str, Any] = {
        "outcome": status, "status": status,
        "claim_ids": CLAIM_IDS, "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": "non_contributory",
        "supersedes": SUPERSEDES, "sleep_driver_pattern": "N/A (no sleep loop)",
        "hypothesis_space": {"qid": "contextmemory_write_content_discrimination",
                             "hid": "H1-loss-objective-mismatch"},
        "metrics": metrics, "arm_results": rows_a + rows_b,
        "summary_markdown": "\n".join(lines),
        "per_arm_gate": gate["per_arm_gate"], "non_degenerate": h1_any_green,
        "degeneracy_reason": degeneracy_reason,
        "interpretation": {"label": label, "preconditions": adjudication_preconditions,
                           "criteria_non_degenerate": criteria_non_degenerate, "criteria": criteria,
                           "h4_attribution_caveat": h4_caveat},
        "fatal_error_count": 0,
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    if args.dry_run:
        # Smoke assertions: the instrument must be non-trivially engaged.
        for row in result["arm_results"]:
            key = "heldout_real_readout" if row["regime"] == "A" else "synthetic_fresh_readout"
            rd = row[key]
            assert rd["n_draws"] == sum(sum(r) for r in rd["contingency"]), "contingency/draw mismatch"
            assert rd["n_draws"] > 0 and rd["nmi_excess"] == rd["nmi_excess"], (
                f"probe inert in {row['regime']}::{row['arm']}")
            if row["regime"] == "A":
                assert row["heldout_real_readout"]["class_latent_norms"]["dangerous_z_self"] > 0
            if row["objective"] is not None:
                # Positive control of the manipulated path: every trained arm's own
                # objective was monitored at least once in BOTH regimes.
                assert row["objective_convergence"]["n_points"] > 0, (
                    f"trained arm {row['regime']}::{row['arm']} monitored no objective -- "
                    "the manipulated code path never ran at smoke scale")
        print("[smoke] contingency tables recorded for every cell; MI readouts finite where drawn; "
              "every trained arm's objective monitored in both regimes", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "arms": ARMS, "seeds": SEEDS, "lambda_h1": LAMBDA_H1,
        "write_addressing_loss_weight_956": WRITE_ADDRESSING_LOSS_WEIGHT_956,
        "refractory_k": REFRACTORY_K, "training_episodes": TRAINING_EPISODES,
        "heldout_episodes": HELDOUT_EPISODES, "heldout_frac": HELDOUT_FRAC,
        "steps_per_episode": STEPS_PER_EPISODE, "context_switch_every": CONTEXT_SWITCH_EVERY,
        "n_steps_b": N_STEPS_B, "min_heldout_per_class": MIN_HELDOUT_PER_CLASS,
        "min_paired_seeds": MIN_PAIRED_SEEDS, "step_match_max": STEP_MATCH_MAX,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE, "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR, "num_slots": NUM_SLOTS, "latent_dim": LATENT_DIM,
        "write_calls_floor": WRITE_CALLS_FLOOR, "nmi_margin": NMI_MARGIN,
        "alpha_corrected": ALPHA_CORRECTED, "n_tests": N_TESTS,
        "conv_min_decrease_frac": CONV_MIN_DECREASE_FRAC, "conv_edge_frac": CONV_EDGE_FRAC,
        "probe_clusters": PROBE_CLUSTERS, "probe_jitter": PROBE_JITTER, "probe_n": PROBE_N,
        "cluster_base_scale": CLUSTER_BASE_SCALE, "mi_n_shuffle": MI_N_SHUFFLE,
        "h1_batch_per_side": H1_BATCH_PER_SIDE, "h1_inject_every_n_steps": H1_INJECT_EVERY_N_STEPS,
        "min_h1_buf": MIN_H1_BUF,
    }
    out_path = write_flat_manifest(result, dry_run=args.dry_run, config=full_config, seeds=SEEDS,
                                   script_path=__file__, started_at=t0,
                                   z_goal_stream_stats=zg_accumulator.stats())
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    emit_outcome(outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path, dry_run=args.dry_run)
