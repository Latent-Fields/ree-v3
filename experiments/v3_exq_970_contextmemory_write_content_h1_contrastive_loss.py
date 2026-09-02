#!/opt/local/bin/python3
"""
V3-EXQ-970 -- ContextMemory write-content discrimination: H1 (loss-objective-
mismatch) contrastive-loss probe, real-agent AND synthetic dual-regime, against
the frozen hypothesis-space question `contextmemory_write_content_discrimination`
(registered from the V3-EXQ-956 autopsy).

Chip: chip-20260830-ctxmem-write-content-h1h4-portfolio

experiment_purpose: diagnostic

hypothesis_space:
  qid: contextmemory_write_content_discrimination
  hid: H1-loss-objective-mismatch

H1 (axis: algorithm): "Pairwise addressing-distribution diversity references no
content; a content-referencing (contrastive) objective is required." Design per
the frozen registry entry: swap in a contrastive/cluster-supervised loss at the
same schedule, on BOTH real-agent latents and the synthetic 2-cluster stream as
training input (the dual regime avoids conflation with H4). Null (this
experiment's own falsifier, verbatim): "Jaccard does not drop by the 0.25 margin
in either regime." Precedent: V3-EXQ-907's confirmed read-path result (a
context-referencing objective broke the saddle at the same 15,000-step
schedule); both failed write-side losses (MoE importance; pairwise diversity)
are content-blind objectives.

WHY THIS RUN EXISTS. V3-EXQ-956 measured whether `write_addr_tagger`, trained
via `ContextMemory.compute_write_addressing_loss` (a pairwise-diversity loss
over selection distributions that references NO content label), becomes
content-conditioned. It did not, on the FAIL side of a genuinely open question
-- see 956's own docstring. H1 attributes that failure to the OBJECTIVE, not
the substrate wiring: a loss that never looks at which context produced a
state cannot be expected to learn a context-conditioned selection rule, no
matter how long it trains. This experiment tests the direct fix: replace the
content-blind pairwise-diversity loss with a CONTENT-REFERENCING (contrastive)
one -- reward divergence between the mean selection-probability vectors of two
KNOWN classes (safe/dangerous context in Regime A, cluster id in Regime B) --
computed manually in this driver rather than via
`compute_write_addressing_loss` (which stays at weight=0.0 throughout; see
"SUBSTRATE PROPERTIES" below).

DUAL REGIME, AND WHY BOTH ARE REQUIRED (avoids conflation with H4). H4 (V3-EXQ-
972, sibling in this portfolio) tests whether real-agent train-time latents
carry separable content structure at all. If H1 were tested ONLY in the
real-agent regime and failed, that failure would be ambiguous between "the loss
design does not work" (H1 true) and "the input lacks structure to exploit"
(H4 true, not H1's fault). Regime B trains the IDENTICAL loss on a synthetic
2-cluster stream that has separable content BY CONSTRUCTION (jitter=0.0078 per
cluster, the same instrument 956's own post-training probe uses). If the same
loss succeeds on synthetic data but fails on real-agent data, that is evidence
FOR H4 (input distribution), not against H1 (the objective can shape
content-conditioned selection, given clean input). If it fails on BOTH, that
is genuine evidence against H1 itself.

THE H1 LOSS (shared by both regimes, see `h1_context_divergence_loss` below).
Given two batches of states from two known classes, score both with
`write_addr_tagger`, turn scores into per-example selection-probability
vectors via the SAME "lower score wins" convention already used by
`ContextMemory.compute_write_addressing_loss` (`F.softmax(-scores, dim=-1)`,
see e1_deep.py), take each class's MEAN probability vector, and take the L1
distance between the two means -- exactly the shape of V3-EXQ-907's own
`_context_divergence` (read-path precedent: `agent.e1.cue_slot_tagger`,
`(w_safe - w_dang).abs().sum()` on softmax(logits/temp).mean(dim=0)). This
driver MINIMIZES `-divergence` (a loss to minimize, added into the training
total via `LAMBDA_H1 * h1_loss`) rather than 907's convention of SUBTRACTING
`lambda * divergence` from the total -- the two are mathematically identical
(`total - lambda*divergence == total + lambda*(-divergence)`); stated
explicitly here, once, so the sign convention is unambiguous throughout this
file.

POST-REVIEW FIXES APPLIED (Findings F1-F4, adversarial design review,
2026-09-01). Four fixes to this driver, applied after an earlier draft:

  F1 (Regime B's probe measured generalization to UNSEEN clusters, not
  whether training worked): the old load-bearing probe called
  `_run_content_probe(agent, seed)`, which independently reseeds a
  DIFFERENT pair of synthetic bases via `seed + 500_000` -- correct hygiene
  for Regime A (real-agent latents vs an unrelated synthetic distribution)
  but wrong for Regime B, whose training and probe are BOTH on the same
  synthetic-cluster generative family and are supposed to be the SAME
  clusters. Fixed: `_run_content_probe_on_bases` now probes on the exact
  `bases` the arm trained on (fresh noise draws, same identity), and is
  Regime B's load-bearing readout (`probe_2cluster_jaccard_trained_
  clusters`); the old fresh-bases probe is kept as a secondary,
  non-gating `probe_2cluster_jaccard_fresh_clusters` field.
  F2 (Regime A's probe was on synthetic clusters unrelated to what was
  trained): added a genuinely held-out real-latent probe
  (`_run_heldout_real_latent_probe`) over the LAST N_HELDOUT=200 states of
  each per-context buffer, never sampled into an H1 training batch (see
  `_h1_sample_pool_upper_bound`) -- now Regime A's load-bearing readout
  (`probe_heldout_real_latent_jaccard`); the old generic synthetic probe
  is kept as a secondary, non-gating `probe_2cluster_jaccard_synthetic`
  field.
  F3 (158x update-budget asymmetry between regimes): the H1 loss is now
  injected every `H1_INJECT_EVERY_N_STEPS` (10) steps within an episode
  rather than once per episode boundary -- ~1500 updates total (100
  episodes x 15/episode) vs the old ~95, a ~16x reduction in the asymmetry
  against Regime B's 15000 SGD steps (residual ~10x gap; see the
  `h1_confirmed_synthetic_regime_only_points_to_h4_input_distribution`
  label's caveat below).
  F4 (near-binary per-seed Jaccard, ~15% false-positive rate under a bare
  mean-margin test, no multiplicity correction): a regime's PASS now also
  requires a paired sign-flip permutation-test p-value (`_permutation_
  test_pvalue`) below the Bonferroni-corrected alpha (0.05 / 2 regimes =
  0.025); the old bare-margin-only reading is kept as a secondary,
  non-gating `H1_bare_margin_reading_non_gating` criterion.

red-team (fable), 2026-09-02: CONTESTED -> F5 FIXED. At the F4 fix's own
exact sign-flip enumeration, n=5 seeds made the minimum ATTAINABLE
one-sided p-value 1/2**5 = 0.03125, exceeding ALPHA_CORRECTED (0.025) --
the load-bearing PASS branch was unreachable under EVERY outcome, including
a perfect same-direction result on all 5 seeds. Fixed by adding a 6th seed
(300) to `SEEDS`: 1/2**6 = 0.015625 < 0.025. See Finding F5 at
`_permutation_test_pvalue`'s own docstring for the full arithmetic. No
other red-team finding survived verification against the source for this
script.

REGIME A (real-agent): reuses V3-EXQ-956's FULL harness (env, agent, episode
loop) essentially unchanged, with `contextmemory_write_addressing_loss_weight`
pinned at 0.0 for EVERY arm (the substrate's own pairwise-diversity loss is
NEVER added by `agent.compute_prediction_loss()`; H1's own loss is computed
and injected manually in this driver's own copy of the training step). Two
arms:
  REGIME_A_UNTRAINED   H1 loss weight = 0.0 (write_addr_tagger never receives
                        ANY gradient -- a fresh baseline; NOT a reuse of
                        956's own GUMBEL_UNTRAINED cells: different driver,
                        different fingerprint, mechanical arm-fingerprint
                        reuse is refused for a fresh driver per GOV-REUSE-1 /
                        arm-reuse convention -- see "Step 2.4" below).
  REGIME_A_TRAINED      H1 loss weight = LAMBDA_H1 (0.5, matching 956/907's
                        own weight convention -- see "LAMBDA_H1" below).
Rolling per-context state buffers (`states_safe`/`states_dangerous`, mirrors
956's `harm_buf_pos`/`harm_buf_neg` pattern exactly: same MAX_STATE_BUF=4000
cap, same `del buf[:-MAX_BUF]` eviction) accumulate the write-stream state
tensor (`torch.cat([latent.z_self.detach(), latent.z_world.detach()],
dim=-1)`, matching `ree_core/agent.py`'s own `context_memory.write()` call
site exactly -- confirmed via `grep -n "context_memory.write(" ree_core/
agent.py`), tagged by the CURRENT episode's safe/dangerous label (956's own
`block = ep // CONTEXT_SWITCH_EVERY; is_safe_ep = (block % 2 == 0)`
convention). Once BOTH buffers have >= MIN_H1_BUF (8 -- doubled from 956's own
harm-eval-loss minimum-buffer convention of 4-per-side, since a meaningful
MEAN over a class needs more than the minimum needed for a single BCE step),
a balanced batch (`min(H1_BATCH_PER_SIDE, pool_safe, pool_dangerous)` per
side, via `torch.randperm`, drawn only from each buffer's non-held-out pool
-- `_h1_sample_pool_upper_bound`, Finding F2) is sampled every
`H1_INJECT_EVERY_N_STEPS` (10) steps -- Finding F3 fix, was once per episode
boundary -- and `LAMBDA_H1 * h1_loss` is added into that step's
`total = e1_loss + e2_loss` before `optimizer.step()`, exactly mirroring
956's own per-step training-loss shape (this driver's `_run_episode_a` is
956's `_run_episode` plus this addition and the state-buffer capture; see
956's own docstring "DELIBERATE DIVERGENCE" section for why `agent._e1_tick`
is called every step -- reused unchanged, same reasoning applies). The last
N_HELDOUT=200 states of each buffer are never included in `pool_safe`/
`pool_dangerous` once the buffer has grown past N_HELDOUT + MIN_H1_BUF, so
they remain genuinely held out for the post-training `_run_heldout_real_
latent_probe` (Finding F2's load-bearing readout).
`TRAINING_EPISODES = 100`, `STEPS_PER_EPISODE = 150` (matches V3-EXQ-907's own
real, non-toy 15,000-step schedule -- "same schedule" per the frozen registry
entry's design note; per-episode wall-clock cost is materially unchanged by
the F3 cadence change, since each H1 injection is a cheap forward+backward on
a small batch through a small MLP). Seeds: 956's own `[42, 7, 13, 100, 200]`.

REGIME B (synthetic, no environment): no env, no episode loop -- direct
batched SGD training of `write_addr_tagger` ALONE on the same synthetic
2-cluster generator 956 uses for its EVAL probe (LATENT_DIM=64,
PROBE_JITTER=0.0078, 2 clusters), used here as TRAINING data. A throwaway
`_make_env_safe(seed)` is constructed purely so `_make_agent` can read
`body_obs_dim`/`world_obs_dim`/`action_dim` -- it is never `reset()` or
`step()`ped. `contextmemory_write_addressing_loss_weight=0.0` (same pin as
Regime A; the substrate's own loss is never engaged). Two arms:
  REGIME_B_UNTRAINED   0 SGD steps (tagger at random init).
  REGIME_B_TRAINED      `N_STEPS = 15000` SGD steps (matches V3-EXQ-907's own
                        "same 15,000-step schedule"), each step drawing a
                        fresh balanced batch (`H1_BATCH_PER_SIDE` per cluster)
                        from a per-cell cluster generator (`_regime_b_cluster_
                        bases`/`_regime_b_sample_batch`, the exact same
                        base+jitter*randn construction as 956's own
                        `_probe_stream`, but drawing CONTINUOUSLY from one
                        `torch.Generator` across the whole run instead of a
                        fixed-`n` one-shot list -- so training never repeats
                        the identical sample twice, while remaining
                        apples-to-apples with 956's probe on
                        LATENT_DIM/PROBE_JITTER/PROBE_CLUSTERS), computing
                        `LAMBDA_H1 * h1_context_divergence_loss(...)` alone
                        (no e1_loss/e2_loss -- there is no env here) and
                        stepping a DEDICATED `optim.Adam(write_addr_tagger.
                        parameters(), lr=1e-3)`.
Seeds: the same `[42, 7, 13, 100, 200]` (cheap regime, no reason to economize).
The SAME `bases` (constructed once per cell via `_regime_b_cluster_bases`,
regardless of arm -- REGIME_B_UNTRAINED uses the bases it WOULD have trained
on, for an apples-to-apples untrained baseline) are used by both training
(when n_steps > 0) and the post-training probe. After training (or at init
for UNTRAINED), both arms run TWO eval-mode probes: the LOAD-BEARING
`_run_content_probe_on_bases` on those SAME bases (fresh noise draws,
`probe_2cluster_jaccard_trained_clusters` -- Finding F1's fix), and the
SECONDARY/non-gating `_run_content_probe` on a fresh, unrelated pair of
bases (`seed + 500_000`, `probe_2cluster_jaccard_fresh_clusters` -- "does
this also generalize to an unseen draw of the distribution").

LAMBDA_H1 = 0.5. Not an arbitrary choice: mirrors `sd016_diversification_
weight`'s / 956's own `contextmemory_write_addressing_loss_weight`'s validated
operating point throughout the 418/436/907/956 lineage -- the closest
available precedent for a loss of this pairwise/divergence shape, so 0.5 is
the most defensible single value to sweep at rather than an unjustified pick
(matches 956's own reasoning for the identical number verbatim).

ACCEPTANCE CRITERIA. This experiment uses `experiments._lib.precondition_gate`
(`PreconditionSpec`, `evaluate_arm_gate`, `aggregate_arm_gates`) with each
REGIME (not each arm) as the gated unit, per that module's own "regime-
conditioned" design: a naive `all(...)` gate across two structurally different
regimes would let one regime's structurally-inapplicable precondition vacate
the other regime's valid result (exactly the V3-EXQ-785 failure mode that
module exists to close) -- see the module's own docstring, read in full before
writing this driver.

  PreconditionSpec "writepath_engaged" (P0, regime-A-only): min(n_write_calls
    across all Regime A cells, both arms x all seeds) >= WRITE_CALLS_FLOOR
    (200, 956's own floor). `applies_to`: `regime == "A"` only -- Regime B has
    no environment and no `ContextMemory.write()` calls during training (only
    at the post-training eval probe), so writepath engagement is not a
    meaningful question for it; its own readiness is BY CONSTRUCTION
    (`N_STEPS` is a fixed constant, not empirically gated), stated as the
    `applies_note`.
  PreconditionSpec "untrained_baseline_headroom" (C2-style readiness, BOTH
    regimes): mean_jaccard(<REGIME>_UNTRAINED) >= JACCARD_MARGIN (0.25) --
    each regime's own headroom gate, independent of the other, checked
    empirically rather than assumed (956's own "CORRECTION TO AN EARLIER
    ASSUMPTION" applies here too: an untrained tagger's baseline Jaccard is
    NOT assumed to be near 1.000).

`evaluate_arm_gate("Regime_A", ...)` / `evaluate_arm_gate("Regime_B", ...)`
followed by `aggregate_arm_gates([gate_a, gate_b])` gives `non_degenerate =
any regime green` -- so a Regime A P0 failure alone does NOT fail the whole
run if Regime B's own gate is green, and vice versa (the module's own
semantic fix, applied here across regimes rather than across arms of a single
regime).

LOAD-BEARING CRITERION `H1_content_referencing_objective_succeeds_in_either_
regime` (load_bearing: true). Regime A's Jaccard here is the HELD-OUT
real-latent probe (`probe_heldout_real_latent_jaccard`, Finding F2); Regime
B's is the trained-clusters probe (`probe_2cluster_jaccard_trained_
clusters`, Finding F1):
  PASS iff (Regime A's gate is green AND mean_jaccard(REGIME_A_TRAINED) <=
  mean_jaccard(REGIME_A_UNTRAINED) - JACCARD_MARGIN AND p_value_regime_a <
  ALPHA_CORRECTED) OR (Regime B's gate is green AND
  mean_jaccard(REGIME_B_TRAINED) <= mean_jaccard(REGIME_B_UNTRAINED) -
  JACCARD_MARGIN AND p_value_regime_b < ALPHA_CORRECTED).
  Finding F4 fix: `p_value_regime_<A|B>` is a paired sign-flip permutation
  test (`_permutation_test_pvalue`) over the 6 matched-seed diffs (see
  Finding F5 below for why 5 was not enough) (untrained - trained),
  Bonferroni-corrected across the 2 regimes
  (`ALPHA_CORRECTED = 0.05 / 2 = 0.025`) -- a per-seed Jaccard is
  near-binary, so a bare mean-margin comparison alone has an estimated ~15%
  false-positive rate under the null with no multiplicity correction. The
  OLD bare-margin-only reading is retained as the secondary, non-gating
  `H1_bare_margin_reading_non_gating` criterion.
  combination_rule (recorded verbatim in the manifest, per the multi-criterion
  combination-logic requirement): "OR across regimes -- either regime
  succeeding is sufficient evidence FOR H1; the null (H1 false) requires BOTH
  regimes to fail their own headroom-gated comparison." This prevents a flat
  criteria list with one PASS and one FAIL regime from being misread as
  "mostly failed" -- the gate was always an OR, not an AND.

criteria_non_degenerate: per-regime, keyed on that regime's OWN readiness gate
(`H1_regime_A_content_referencing_objective_succeeds`: Regime A gate green;
`H1_regime_B_content_referencing_objective_succeeds`: Regime B gate green),
per the precondition_gate module's own per-arm(here: per-regime) non-degeneracy
convention.

Both regimes' results are reported DISTINCTLY throughout (per-regime rows,
per-regime metrics, a per-regime summary table) -- never collapsed into one
number. If the regimes DISAGREE (one succeeds, one fails), `interpretation.
label` states plainly which one, since that specific disagreement is itself
informative (real-agent fails + synthetic succeeds points toward H4, the
input-distribution sibling hypothesis, per V3-EXQ-972).

evidence_direction: "non_contributory". claim_ids: [] -- this experiment
tests a SUBSTRATE mechanism's objective design (does a content-referencing
write-addressing loss achieve content-conditioned selection), not a claim
hypothesis; diagnostic-purpose experiments with claim_ids=[] are excluded from
governance confidence scoring by design (matches 956/972's own convention).

Step 2.4 GOV-REUSE-1 (existing-evidence check): the decisive readout (2-cluster
content-discrimination Jaccard for a write_addr_tagger trained under a
CONTENT-REFERENCING objective, in either regime) is not recorded anywhere.
956's own manifest cells trained under the DIFFERENT, content-BLIND pairwise-
diversity loss -- not a substitute measurement for this question -- and were
in any case minted WITHOUT `include_driver_script_in_hash=False`, so even if
the loss were the same, arm-fingerprint reuse would be refused for this
different driver. Not recoverable; proceeding to run fresh.

Step 2.5b re-derive brake: N/A. claim_ids is empty -- no claim-hypothesis
re-test for the brake to apply to.

Step 2.5c substrate-path overlap (re-checked against the CURRENT
substrate_queue.json, 2026-09-01, not merely cited from V3-EXQ-956/972):
every open `corrupting` entry whose `substrate_paths` names `ree_core/
agent.py` or `ree_core/predictors/e1_deep.py` was re-read fresh, same result
V3-EXQ-972 (this same worktree, same day) independently found:
  - `contextmemory-write-path-addressing-degeneracy` (e1_deep.py::
    ContextMemory.write, status implemented_pending_validation) -- this IS
    the entry this experiment's write-stream is drawn from (Regime A) / the
    mechanism under test (both regimes); not an unrelated overlap.
  - `modulatory-bias-selection-authority` (agent.py::REEAgent.select_action,
    status implemented) -- closed/non-blocking.
  - `mode-governance-engagement` (agent.py, whole-file, status
    implemented_pending_validation) -- removed by construction: this driver
    never sets `use_salience_coordinator=True` (REEConfig's own False default
    used throughout) and never reads `agent.operating_mode` or any
    SalienceCoordinator output.
  - `SD-082` (agent.py, whole-file, status implemented_pending_validation) --
    removed by construction: `lateral_pfc_rule_readout_consumer` is a no-op
    default per its own implementation_note_update; this driver never sets
    `use_lateral_pfc_analog=True` and never enables the SD-078 rule-pool /
    SD-082 consumer path.
  - `SD-e1-rollout-consistency-training` (e1_deep.py::forward,
    e1_deep.py::predict_long_horizon, status item1_validated_item2_pending)
    -- removed by construction: this driver never calls `predict_long_
    horizon` (no multi-step rollout, in either regime) and never enables an
    action-conditioned / rollout-consistency training objective; Regime A
    exercises only the ordinary single-step `agent.sense()` -> E1 forward
    path (unaffected by ITEM 2's still-pending status) and Regime B never
    calls `agent.sense()` at all.
  - `SD-056` no longer names agent.py in the current queue (e2_fast.py /
    e3_selector.py / config.py only) -- non-blocking either way.
No other open `corrupting` entry's substrate_paths overlaps a module this
driver imports or exercises.

Does NOT flip the `contextmemory-write-path-addressing-degeneracy` substrate_
queue entry's status regardless of outcome -- that stays a human/governance
disposition, unchanged from 956/972's own convention.

SUBSTRATE PROPERTIES held constant across both regimes/all arms (not part of
the manipulation), identical to V3-EXQ-956/972's own list, reused verbatim:
  - contextmemory_gated_content_write=True (436d write-path repair).
  - sd016_writepath_mode="sense_only".
  - alpha_world=0.9 (SD-008).
  - use_noise_floor=True (MECH-313/ARC-065).
  - context_memory.memory.requires_grad_(False) after construction (436e/f
    Adam-drift neutralization). write_addr_tagger is deliberately left
    trainable -- it is this experiment's own manipulated parameter.
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True.
  - use_salience_coordinator / use_coalition_controller left at REEConfig
    defaults (both False) -- see Step 2.5c above.
  - sd016_enabled left at its REEConfig default (False).
  - contextmemory_write_addressing_loss_weight = 0.0 for EVERY cell in EVERY
    regime, always -- H1's own loss is computed and injected manually by this
    driver, never via the substrate's own `compute_write_addressing_loss`
    path (that path IS the content-blind objective this experiment replaces).

No sleep. SD-016 sleep-cycle consolidation is orthogonal to whether the
write-address objective itself can be made content-conditioned.
sleep_driver_pattern: "N/A (no sleep loop)"

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow
"""

import argparse
import itertools
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec, evaluate_arm_gate, aggregate_arm_gates,
)


EXPERIMENT_TYPE = "v3_exq_970_contextmemory_write_content_h1_contrastive_loss"
QUEUE_ID = "V3-EXQ-970"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: List[int] = [42, 7, 13, 100, 200, 300]  # 956/943/907/972's 5 plus one
# more (300, matching the sibling H2 script's own Phase B addition) -- see
# Finding F5 below: n=5 made the load-bearing permutation test's PASS branch
# unreachable regardless of true effect size.

LAMBDA_H1 = 0.5  # see module docstring "LAMBDA_H1"

# Regime A arms: (label, h1_loss_weight)
ARMS_A: List[Tuple[str, float]] = [
    ("REGIME_A_UNTRAINED", 0.0),
    ("REGIME_A_TRAINED", LAMBDA_H1),
]
# Regime B arms: (label, n_sgd_steps)
N_STEPS_B = 15000  # matches V3-EXQ-907's own real 15,000-step schedule
ARMS_B: List[Tuple[str, int]] = [
    ("REGIME_B_UNTRAINED", 0),
    ("REGIME_B_TRAINED", N_STEPS_B),
]

# Substrate properties held constant across all cells, all regimes.
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0
WRITE_SELECTION = "gumbel_learned"
WRITE_ADDRESSING_LOSS_WEIGHT = 0.0  # H1's own loss is injected manually -- see docstring

TRAINING_EPISODES = 100  # Regime A -- matches V3-EXQ-907/956's own real schedule
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000
MAX_STATE_BUF = 4000  # per-context write-stream state buffer cap (Regime A)

MIN_H1_BUF = 8  # doubled from 956's own harm-eval-loss 4-per-side minimum
H1_BATCH_PER_SIDE = 32  # balanced-batch size per class for the H1 loss

# Finding F3 fix: inject the H1 loss every N steps within an episode rather
# than once per episode boundary, to shrink the Regime A vs Regime B
# update-budget asymmetry (95 updates -> ~1500 updates; still ~10x fewer
# than Regime B's 15000, see module docstring "REGIME A" and the
# `h1_confirmed_synthetic_regime_only_points_to_h4_input_distribution` label).
H1_INJECT_EVERY_N_STEPS = 10

# Finding F2 fix: the last N_HELDOUT states appended to each per-context
# state buffer are reserved as a genuinely-never-trained-on tail for
# Regime A's load-bearing held-out real-latent probe (see
# `_h1_sample_pool_upper_bound` / `_run_heldout_real_latent_probe` below).
N_HELDOUT = 200

# Finding F4 fix: Bonferroni correction across the 2 regimes' permutation
# tests (Regime A, Regime B), each tested at the family-wise 0.05 level.
ALPHA_CORRECTED = 0.05 / 2

NUM_SLOTS = 16  # ContextMemory default
LATENT_DIM = 64  # self_dim(32) + world_dim(32) -- must match the contract
                 # test's own LATENT_DIM for the probes to be like-for-like.

# P0 readiness floor (Regime A only) -- exactly V3-EXQ-956/943's own.
WRITE_CALLS_FLOOR = 200.0

# C2-style readiness / directional-drop margin (both regimes).
JACCARD_MARGIN = 0.25

# Post-training / Regime-B-training synthetic 2-cluster instrument (reuses
# 956's own instrument -- test_contextmemory_write_address_selection.py's
# _stream/_jaccard).
PROBE_N = 1500
PROBE_JITTER = 0.0078
PROBE_CLUSTERS = 2
CLUSTER_BASE_SCALE = 0.078  # matches 956's own _probe_stream base scale

PRINT_EVERY_B = 2000  # Regime B training progress cadence (full-scale)

TAGGER_MOVE_EPS = 1e-8  # threshold for "did write_addr_tagger's state_dict move"


# ------------------------------------------------------------------ #
# H1's own loss -- shared by both regimes. See module docstring "THE H1 LOSS".
# ------------------------------------------------------------------ #

def h1_context_divergence_loss(write_addr_tagger, states_a: torch.Tensor,
                                states_b: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Content-referencing (contrastive) write-addressing loss. Scores both
    class batches with `write_addr_tagger`, converts to per-example
    selection-probability vectors via the SAME "lower score wins" convention
    `ContextMemory.compute_write_addressing_loss` uses
    (`F.softmax(-scores, dim=-1)`), takes each class's MEAN probability
    vector, and returns the NEGATIVE L1 distance between the two means --
    mirrors V3-EXQ-907's own `_context_divergence` (read-path precedent) in
    shape (softmax convention, L1 distance, mean-then-distance ordering), but
    expressed as a quantity to MINIMIZE (a loss added into the training total
    via `LAMBDA_H1 * h1_loss`) rather than 907's convention of subtracting
    `lambda * divergence` from the total -- the two are mathematically
    identical.
    """
    scores_a = write_addr_tagger(states_a)
    scores_b = write_addr_tagger(states_b)
    probs_a = F.softmax(-scores_a / tau, dim=-1).mean(dim=0)
    probs_b = F.softmax(-scores_b / tau, dim=-1).mean(dim=0)
    divergence = (probs_a - probs_b).abs().sum()
    return -divergence


# ------------------------------------------------------------------ #
# Finding F4 -- paired sign-flip permutation test.                          #
# ------------------------------------------------------------------ #

def _permutation_test_pvalue(diffs: List[float], n_perm: int = 100_000,
                              perm_seed: int = 0) -> float:
    """Paired sign-flip permutation test (one-sided), same statistical
    approach applied to the sibling H2 script
    (v3_exq_969_contextmemory_write_content_h2_operating_point.py) for the
    identical per-seed-Jaccard-margin testing problem -- adapted here as
    this file's own copy (the sibling script did not expose an importable
    `_permutation_test_pvalue` helper at the time this fix was written, so
    this is a from-scratch implementation of the same method, not a literal
    copy) since a bare mean-margin comparison over near-binary per-seed
    Jaccards has an estimated ~15% false-positive rate under the null with
    no multiplicity correction (Finding F4).

    Under the null (no true directional difference between the TRAINED and
    UNTRAINED arms for a given seed), each paired diff's sign is
    exchangeable, so every sign assignment is equally likely. For n<=20
    pairs (this file uses n=6, one per seed -- see Finding F5 below for why
    5 was not enough) every one of the 2**n sign patterns is enumerated
    EXACTLY -- cheap, and avoids any Monte-Carlo noise. For a hypothetically
    larger n this falls back to `n_perm` Monte-Carlo sign-flip draws under a
    fixed `perm_seed`, for reproducibility.

    Finding F5 (adversarial design review, fable, 2026-09-02): at the exact
    sign-flip test's own arithmetic, n=5 seeds makes the minimum ATTAINABLE
    one-sided p-value 1/2**5 = 0.03125 -- greater than ALPHA_CORRECTED
    (0.025) -- so the load-bearing PASS branch was unreachable under EVERY
    possible outcome, including a perfect 5/5 same-direction result. n=6
    lowers the floor to 1/2**6 = 0.015625 < 0.025, restoring genuine
    two-sided discriminability. Both regimes share `SEEDS`, so the fix
    applies to both.

    Returns the one-sided p-value: the fraction of permutations whose mean
    diff is >= the observed mean diff (ties counted as extreme -- the
    standard conservative convention), i.e. "how often would a random sign
    assignment produce a drop this large or larger, by chance alone".
    `diffs` should be `untrained_i - trained_i` per matched seed, so a
    large positive mean is evidence FOR H1 (trained arm's Jaccard dropped).
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


def _paired_diffs(rows_untrained: List[Dict[str, Any]],
                   rows_trained: List[Dict[str, Any]],
                   value_key: str) -> List[float]:
    """Matches UNTRAINED/TRAINED rows by seed (both arms iterate SEEDS in
    the same order, but matching by seed rather than by list position is
    defensive against that assumption ever changing) and returns
    `untrained - trained` per seed, skipping any seed where either side's
    value is None (e.g. Regime A's held-out probe when a seed's buffer
    never reached N_HELDOUT + MIN_H1_BUF -- see Finding F2)."""
    trained_by_seed = {r["seed"]: r.get(value_key) for r in rows_trained}
    diffs: List[float] = []
    for r in rows_untrained:
        u = r.get(value_key)
        t = trained_by_seed.get(r["seed"])
        if u is None or t is None:
            continue
        diffs.append(u - t)
    return diffs


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused verbatim from V3-EXQ-956/943/972)  #
# ------------------------------------------------------------------ #

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=1,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.10,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=10,
        num_hazards=8,
        num_resources=4,
        hazard_harm=0.05,
        env_drift_interval=50,
        env_drift_prob=0.05,
        proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """Always write_selection='gumbel_learned',
    contextmemory_write_addressing_loss_weight=0.0 -- H1's own loss is
    injected manually by this driver (Regime A: _run_episode_a; Regime B:
    _train_regime_b), never via the substrate's own
    compute_write_addressing_loss path. Reused for both regimes; Regime B
    passes a throwaway, never-reset/never-stepped env purely for the dim
    kwargs.
    """
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
        contextmemory_write_selection=WRITE_SELECTION,
        contextmemory_write_addressing_loss_weight=WRITE_ADDRESSING_LOSS_WEIGHT,
        use_noise_floor=USE_NOISE_FLOOR,
        noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None, (
        "use_noise_floor=True did not construct agent.noise_floor -- "
        "REEConfig/REEAgent wiring regression."
    )
    assert agent.e1.context_memory.num_slots == NUM_SLOTS, (
        f"ContextMemory num_slots changed from the assumed {NUM_SLOTS} -- "
        "update NUM_SLOTS before trusting downstream thresholds."
    )
    total_latent_dim = agent.e1.config.self_dim + agent.e1.config.world_dim
    assert total_latent_dim == LATENT_DIM, (
        f"self_dim+world_dim={total_latent_dim} != LATENT_DIM={LATENT_DIM} -- "
        "the synthetic-stream / probe instruments would no longer match; "
        "update LATENT_DIM before trusting the H1 loss or the probes."
    )
    assert agent.e1.context_memory.write_addr_tagger is not None, (
        "write_selection='gumbel_learned' did not construct write_addr_tagger "
        "-- config wiring regression."
    )
    # 436e/f Adam-drift neutralization. write_addr_tagger is deliberately NOT
    # frozen -- it is this experiment's own manipulated parameter.
    agent.e1.context_memory.memory.requires_grad_(False)
    return agent


def _action_onehot(a_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, a_idx] = 1.0
    return v


def _effective_temperature(agent: REEAgent) -> float:
    if agent.noise_floor is not None:
        return agent.noise_floor.compute_effective_temperature(
            baseline_temperature=BASELINE_TEMPERATURE, simulation_mode=False,
        )
    return BASELINE_TEMPERATURE


def _select_action_baseline(agent: REEAgent, z_world: torch.Tensor,
                             num_actions: int) -> int:
    """Temperature-graded softmax sample over predicted harm, reused verbatim
    from the 436/943/956/972 lineage's baseline policy.
    """
    with torch.no_grad():
        harms: List[float] = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            zw_next = agent.e2.world_forward(z_world, a_oh)
            h = agent.e3.harm_eval(zw_next).mean().item()
            harms.append(h)
        eff_t = _effective_temperature(agent)
        harms_t = torch.tensor(harms, dtype=torch.float32)
        probs = F.softmax(-harms_t / eff_t, dim=0)
        best_a = int(torch.multinomial(probs, 1).item())
    return best_a


# ------------------------------------------------------------------ #
# Regime A write-address instrumentation (P0 readiness only -- Regime A's own #
# DV is the post-training probe, not occupancy). Identical to V3-EXQ-956.    #
# ------------------------------------------------------------------ #

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
                assert n_new == 1, (
                    f"expected at most one write() per sense() call, got "
                    f"{n_new} new writes in one poll -- polling cadence "
                    f"assumption violated, sequence would be ambiguous"
                )
                self.sequence.append(int(idx))
            self._last_total = total


def _n_write_calls(sequence: List[int]) -> int:
    return len(sequence)


# ------------------------------------------------------------------ #
# Post-training content-discrimination probe (copied verbatim from 956/972). #
# ------------------------------------------------------------------ #

def _probe_stream(seed: int, n: int, latent_dim: int, jitter: float, clusters: int):
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, latent_dim, generator=gen) * CLUSTER_BASE_SCALE for _ in range(clusters)]
    return [
        (i % clusters, bases[i % clusters] + torch.randn(1, latent_dim, generator=gen) * jitter)
        for i in range(n)
    ]


def _jaccard(per_cluster: Dict[int, Set[int]]) -> float:
    a, b = per_cluster.get(0, set()), per_cluster.get(1, set())
    return len(a & b) / max(len(a | b), 1)


def _run_content_probe(agent: REEAgent, seed: int) -> Tuple[float, Dict[int, Set[int]]]:
    """Post-training, EVAL-MODE probe on a FRESH, unrelated pair of synthetic
    cluster bases (`seed + 500_000` reseeds new bases via `_probe_stream`,
    independently of whatever bases the arm was actually trained on). eval()
    makes gumbel_learned selection deterministic argmin on write_addr_
    tagger's own (possibly-trained) scores. Probe-specific seed offset
    (+500_000) so this measurement's RNG stream never collides with
    training's own consumption.

    NON-GATING / DESCRIPTIVE ONLY (Findings F1 and F2). For Regime A this is
    "does the trained tagger also generalize to an unrelated synthetic
    distribution" -- secondary to the real held-out safe/dangerous-latent
    probe (`_run_heldout_real_latent_probe`), which is Regime A's own
    load-bearing readout. For Regime B this is "does the trained tagger
    also generalize to a wholly different draw of the underlying
    distribution (different cluster bases)" -- secondary to
    `_run_content_probe_on_bases`, which evaluates on the SAME bases
    Regime B actually trained on and is Regime B's own load-bearing
    readout. Reusing this generic (fresh-bases) probe as the LOAD-BEARING
    readout for Regime B was the bug Finding F1 fixes: Regime B trains and
    evaluates on the same synthetic-cluster generative family and the two
    are SUPPOSED to be the same clusters, so a fresh-bases probe measures
    generalization to unseen clusters, not whether training worked at all.
    """
    agent.eval()
    context_memory = agent.e1.context_memory
    per_cluster: Dict[int, Set[int]] = {}
    with torch.no_grad():
        for cid, state in _probe_stream(
            seed + 500_000, PROBE_N, LATENT_DIM, PROBE_JITTER, PROBE_CLUSTERS
        ):
            context_memory.write(state)
            per_cluster.setdefault(cid, set()).add(int(context_memory.last_write_index))
    return _jaccard(per_cluster), per_cluster


def _run_content_probe_on_bases(agent: REEAgent, bases: List[torch.Tensor],
                                 jitter: float, n: int,
                                 gen_seed: int) -> Tuple[float, Dict[int, Set[int]]]:
    """Finding F1 fix -- Regime B's LOAD-BEARING probe. Identical mechanics
    to `_run_content_probe` (agent.eval(), torch.no_grad(), deterministic
    argmin selection via context_memory.write(), record which slot each
    state routes to, Jaccard over occupied-slot sets) but drawn from the
    SAME cluster `bases` the arm was actually trained on (Regime B's
    UNTRAINED arm uses the bases it WOULD have trained on, for an
    apples-to-apples untrained-baseline comparison -- see `_run_cell_b`),
    with FRESH noise draws from a generator seeded independently of
    training's own continuous generator (so this is not literally
    re-scoring the training batches -- still an honest
    OOD-noise-but-same-cluster-identity probe).
    """
    agent.eval()
    context_memory = agent.e1.context_memory
    gen = torch.Generator().manual_seed(gen_seed)
    clusters = len(bases)
    per_cluster: Dict[int, Set[int]] = {}
    with torch.no_grad():
        for i in range(n):
            cid = i % clusters
            latent_dim = bases[cid].shape[-1]
            state = bases[cid] + torch.randn(1, latent_dim, generator=gen) * jitter
            context_memory.write(state)
            per_cluster.setdefault(cid, set()).add(int(context_memory.last_write_index))
    return _jaccard(per_cluster), per_cluster


# ------------------------------------------------------------------ #
# REGIME A -- real-agent episode/cell runners.                              #
# ------------------------------------------------------------------ #

def _h1_sample_pool_upper_bound(buf_len: int) -> int:
    """Finding F2 fix. Once a per-context state buffer has grown past
    N_HELDOUT + MIN_H1_BUF, exclude its own last N_HELDOUT entries from the
    H1-loss training-sampling pool so they remain genuinely held out for
    the post-training real-latent probe (`_run_heldout_real_latent_probe`).
    Below that threshold there is not enough data to reserve a held-out
    tail without starving the H1-loss batch itself, so sampling still draws
    from the full buffer -- that seed's held-out probe is then marked
    insufficient (`probe_heldout_insufficient=True`) rather than silently
    treating those early states as held out when they were not.

    Correctness argument (see the F2 fix report): any state that ends up
    among the FINAL N_HELDOUT entries of the buffer, by definition, never
    has N_HELDOUT-or-more states appended after it at ANY point during the
    run (the count of "states appended after it" only grows monotonically
    over time, and its final value is < N_HELDOUT). So at every earlier
    sampling event too, it was within the excluded last-N_HELDOUT window --
    i.e. it was NEVER eligible to be drawn, at any point, not just at the
    end. `_run_cell_a` additionally asserts this by tracking sampled tensor
    identities and checking them against the final held-out tail.
    """
    return buf_len - N_HELDOUT if buf_len > (N_HELDOUT + MIN_H1_BUF) else buf_len


def _run_heldout_real_latent_probe(
    agent: REEAgent, heldout_safe: List[torch.Tensor],
    heldout_dangerous: List[torch.Tensor],
) -> Tuple[Optional[float], Dict[str, List[int]]]:
    """Finding F2 fix -- Regime A's LOAD-BEARING readout. Mirrors
    `_run_content_probe`'s own mechanics (agent.eval(), torch.no_grad(),
    deterministic argmin selection via context_memory.write(), record which
    slot each state routes to) but on REAL held-out safe/dangerous latents
    -- the actual thing write_addr_tagger was trained to discriminate --
    rather than an unrelated synthetic 2-cluster distribution. Jaccard is
    computed between the safe-held-out occupied-slot set and the
    dangerous-held-out occupied-slot set (low Jaccard = the tagger routes
    safe and dangerous latents to disjoint slots -- content-conditioned
    selection).

    Returns `(None, {"safe": [], "dangerous": []})` when either held-out
    set is empty -- caller marks this `probe_heldout_insufficient=True` and
    excludes the seed from the load-bearing mean/permutation test rather
    than fabricating a number from data that either wasn't held out at all
    or doesn't exist yet (e.g. a --dry-run smoke test's tiny buffers).
    """
    if not heldout_safe or not heldout_dangerous:
        return None, {"safe": [], "dangerous": []}
    agent.eval()
    context_memory = agent.e1.context_memory
    occ_safe: Set[int] = set()
    occ_dangerous: Set[int] = set()
    with torch.no_grad():
        for state in heldout_safe:
            context_memory.write(state)
            occ_safe.add(int(context_memory.last_write_index))
        for state in heldout_dangerous:
            context_memory.write(state)
            occ_dangerous.add(int(context_memory.last_write_index))
    jaccard = len(occ_safe & occ_dangerous) / max(len(occ_safe | occ_dangerous), 1)
    return jaccard, {"safe": sorted(occ_safe), "dangerous": sorted(occ_dangerous)}

def _run_episode_a(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List[torch.Tensor],
    harm_buf_neg: List[torch.Tensor],
    states_safe: List[torch.Tensor],
    states_dangerous: List[torch.Tensor],
    is_safe_ep: bool,
    tracker: _WriteSequenceTracker,
    lambda_h1: float,
    sampled_state_ids: Set[int],
) -> Optional[float]:
    """956's own episode runner (sense -> E1 tick -> baseline action ->
    step -> prediction-loss training -> harm_eval training), PLUS: (1) the
    write-stream state tensor is captured every step into the per-context
    buffer (972's own addition, reused verbatim), and (2) every
    H1_INJECT_EVERY_N_STEPS steps (Finding F3 fix -- was once per episode
    boundary; see module docstring "REGIME A"), if lambda_h1 > 0 and both
    buffers have reached MIN_H1_BUF, a balanced batch is sampled -- drawn
    only from each buffer's non-held-out pool per `_h1_sample_pool_upper_
    bound` (Finding F2 fix) -- and `LAMBDA_H1 * h1_context_divergence_loss
    (...)` is added into that step's total loss before optimizer.step().
    `sampled_state_ids` accumulates `id()` of every individual state tensor
    ever drawn into an H1 batch, across the whole cell -- `_run_cell_a`
    asserts this never intersects the final held-out tail (Finding F2
    correctness check).
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    state_buf = states_safe if is_safe_ep else states_dangerous
    h1_loss_value: Optional[float] = None

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

        # State-tensor capture, exact write() call-site construction.
        obs_state = torch.cat(
            [latent.z_self.detach(), latent.z_world.detach()], dim=-1
        ).cpu()
        state_buf.append(obs_state)
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

        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total = e1_loss + e2_loss

        should_inject_h1 = (
            lambda_h1 > 0.0
            and (_step + 1) % H1_INJECT_EVERY_N_STEPS == 0
            and len(states_safe) >= MIN_H1_BUF
            and len(states_dangerous) >= MIN_H1_BUF
        )
        if should_inject_h1:
            # Finding F2: sample only from the non-held-out pool of each
            # buffer, so the true tail (last N_HELDOUT states) is never
            # drawn into a training batch.
            pool_safe = _h1_sample_pool_upper_bound(len(states_safe))
            pool_dang = _h1_sample_pool_upper_bound(len(states_dangerous))
            k = min(H1_BATCH_PER_SIDE, pool_safe, pool_dang)
            safe_idx = torch.randperm(pool_safe)[:k].tolist()
            dang_idx = torch.randperm(pool_dang)[:k].tolist()
            sampled_state_ids.update(id(states_safe[i]) for i in safe_idx)
            sampled_state_ids.update(id(states_dangerous[i]) for i in dang_idx)
            batch_safe = torch.cat([states_safe[i] for i in safe_idx], dim=0)
            batch_dang = torch.cat([states_dangerous[i] for i in dang_idx], dim=0)
            h1_loss = h1_context_divergence_loss(
                agent.e1.context_memory.write_addr_tagger, batch_safe, batch_dang
            )
            total = total + lambda_h1 * h1_loss
            h1_loss_value = float(h1_loss.item())

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
            zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
            zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
            zw_b = torch.cat([zw_pos, zw_neg], dim=0)
            target_t = torch.cat([
                torch.ones(k_pos, 1, device=agent.device),
                torch.zeros(k_neg, 1, device=agent.device),
            ], dim=0)
            pred = agent.e3.harm_eval_head(zw_b)
            h_loss = F.binary_cross_entropy_with_logits(pred, target_t)
            harm_eval_opt.zero_grad()
            h_loss.backward()
            harm_eval_opt.step()

        if done:
            break

    return h1_loss_value


def _run_cell_a(arm_name: str, lambda_h1: float, seed: int,
                 base_config_slice: Dict[str, Any], zg: ZGoalStreamAccumulator,
                 training_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "regime": "A",
        "arm": arm_name,
        "lambda_h1": lambda_h1,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe)

        tagger_init_state = {
            k: v.clone()
            for k, v in agent.e1.context_memory.write_addr_tagger.state_dict().items()
        }

        standard_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "context_memory.memory" not in n
        ]
        harm_eval_params = list(agent.e3.harm_eval_head.parameters())
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        states_safe: List[torch.Tensor] = []
        states_dangerous: List[torch.Tensor] = []
        tracker = _WriteSequenceTracker()
        sampled_state_ids: Set[int] = set()

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        last_h1_loss: Optional[float] = None
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            h1v = _run_episode_a(
                agent, env, steps_per_episode, optimizer, harm_eval_opt,
                harm_buf_pos, harm_buf_neg, states_safe, states_dangerous,
                is_safe_ep, tracker, lambda_h1, sampled_state_ids,
            )
            if h1v is not None:
                last_h1_loss = h1v
            if (ep + 1) % 50 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] regime=A arm={arm_name} seed={seed} "
                    f"ep {ep + 1}/{training_episodes} "
                    f"n_writes={_n_write_calls(tracker.sequence)} "
                    f"n_states_safe={len(states_safe)} "
                    f"n_states_dangerous={len(states_dangerous)} "
                    f"last_h1_loss={last_h1_loss}",
                    flush=True,
                )

        zg.observe(agent)
        n_write_calls = _n_write_calls(tracker.sequence)

        final_state = agent.e1.context_memory.write_addr_tagger.state_dict()
        max_abs_diff = 0.0
        for k, v0 in tagger_init_state.items():
            d = (final_state[k] - v0).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
        tagger_params_moved = max_abs_diff > TAGGER_MOVE_EPS

        # Finding F2 fix -- held-out real-latent probe, Regime A's own
        # LOAD-BEARING readout. Only the last N_HELDOUT states of each
        # buffer are used, and only once the buffer grew past
        # N_HELDOUT + MIN_H1_BUF (see `_h1_sample_pool_upper_bound`).
        heldout_safe = (
            states_safe[-N_HELDOUT:]
            if len(states_safe) > (N_HELDOUT + MIN_H1_BUF) else []
        )
        heldout_dangerous = (
            states_dangerous[-N_HELDOUT:]
            if len(states_dangerous) > (N_HELDOUT + MIN_H1_BUF) else []
        )
        probe_heldout_insufficient = (not heldout_safe) or (not heldout_dangerous)

        # Correctness check for point (b) of the F2 fix: the held-out tail
        # must never have been drawn into an H1 training batch.
        _heldout_ids = {id(s) for s in heldout_safe} | {id(s) for s in heldout_dangerous}
        assert not (_heldout_ids & sampled_state_ids), (
            "Finding F2 invariant violated: the held-out probe tail "
            "overlaps a state that was sampled into an H1 training batch "
            f"(regime=A arm={arm_name} seed={seed})."
        )

        probe_heldout_jaccard, probe_heldout_occupied = _run_heldout_real_latent_probe(
            agent, heldout_safe, heldout_dangerous
        )

        # Secondary / non-gating (Finding F2): generalization to an
        # unrelated synthetic 2-cluster distribution.
        probe_synthetic_jaccard, probe_synthetic_per_cluster = _run_content_probe(agent, seed)

        row: Dict[str, Any] = {
            "regime": "A",
            "arm": arm_name,
            "seed": seed,
            "lambda_h1": lambda_h1,
            "n_write_calls": n_write_calls,
            "n_states_safe_raw": len(states_safe),
            "n_states_dangerous_raw": len(states_dangerous),
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_heldout_real_latent_jaccard": probe_heldout_jaccard,
            "probe_heldout_real_latent_occupied": probe_heldout_occupied,
            "probe_heldout_insufficient": probe_heldout_insufficient,
            "n_heldout_safe": len(heldout_safe),
            "n_heldout_dangerous": len(heldout_dangerous),
            "probe_2cluster_jaccard_synthetic": probe_synthetic_jaccard,
            "probe_2cluster_occupied_synthetic": {
                str(cid): sorted(slots) for cid, slots in probe_synthetic_per_cluster.items()
            },
        }
        cell.stamp(row)

    print(
        f"verdict: {'PASS' if n_write_calls >= WRITE_CALLS_FLOOR else 'FAIL'} "
        f"(regime=A writepath engagement; n_write_calls={n_write_calls}, "
        f"probe_heldout_jaccard={probe_heldout_jaccard}, "
        f"probe_synthetic_jaccard={probe_synthetic_jaccard:.3f})",
        flush=True,
    )
    return row


# ------------------------------------------------------------------ #
# REGIME B -- synthetic 2-cluster SGD training, no environment.             #
# ------------------------------------------------------------------ #

def _regime_b_cluster_bases(seed: int, latent_dim: int, clusters: int):
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, latent_dim, generator=gen) * CLUSTER_BASE_SCALE for _ in range(clusters)]
    return bases, gen


def _regime_b_sample_batch(bases: List[torch.Tensor], gen: torch.Generator,
                            batch_per_cluster: int, latent_dim: int,
                            jitter: float, clusters: int) -> Dict[int, torch.Tensor]:
    """Draws a fresh batch per cluster CONTINUOUSLY from `gen` (never repeats
    the identical sample twice across a run), using the exact same
    base + jitter*randn construction as 956's own `_probe_stream` -- so
    training and the post-training probe share the same generative model,
    just sampled differently (a fixed n-length list there, a continuous
    stream here).
    """
    out: Dict[int, torch.Tensor] = {}
    for cid in range(clusters):
        noise = torch.randn(batch_per_cluster, latent_dim, generator=gen) * jitter
        out[cid] = bases[cid].expand(batch_per_cluster, -1) + noise
    return out


def _train_regime_b(write_addr_tagger, bases: List[torch.Tensor], gen: torch.Generator,
                     n_steps: int, seed: int, print_every: int) -> Optional[float]:
    """Finding F1 fix: `bases` (and the continuous training-sampling `gen`)
    are now constructed by the CALLER (`_run_cell_b`, via
    `_regime_b_cluster_bases(seed, ...)`, no offset) and passed in, so the
    same object is available afterward for the load-bearing post-training
    probe (`_run_content_probe_on_bases`) to evaluate on the SAME bases
    this function actually trained on -- previously the post-training probe
    independently reseeded a DIFFERENT, unrelated pair of bases via
    `seed + 500_000`, which measures generalization to unseen clusters, not
    whether training worked at all.
    """
    if n_steps <= 0:
        return None
    optimizer = optim.Adam(write_addr_tagger.parameters(), lr=1e-3)
    last_loss = None
    for step in range(n_steps):
        batches = _regime_b_sample_batch(
            bases, gen, H1_BATCH_PER_SIDE, LATENT_DIM, PROBE_JITTER, PROBE_CLUSTERS
        )
        h1_loss = h1_context_divergence_loss(write_addr_tagger, batches[0], batches[1])
        scaled = LAMBDA_H1 * h1_loss
        optimizer.zero_grad()
        scaled.backward()
        optimizer.step()
        last_loss = float(h1_loss.item())
        if (step + 1) % print_every == 0 or (step + 1) == n_steps:
            print(
                f"  [train] regime=B seed={seed} step {step + 1}/{n_steps} "
                f"h1_loss={last_loss:.4f}",
                flush=True,
            )
    return last_loss


def _run_cell_b(arm_name: str, n_steps: int, seed: int,
                 base_config_slice: Dict[str, Any], zg: ZGoalStreamAccumulator,
                 print_every: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "regime": "B",
        "arm": arm_name,
        "n_steps": n_steps,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        # Throwaway env -- dims only, never reset()/step()ped.
        env_throwaway = _make_env_safe(seed)
        agent = _make_agent(env_throwaway)

        tagger_init_state = {
            k: v.clone()
            for k, v in agent.e1.context_memory.write_addr_tagger.state_dict().items()
        }

        # Finding F1 fix: construct the cluster bases ONCE, here, regardless
        # of arm -- REGIME_B_UNTRAINED (0 SGD steps) uses the bases it WOULD
        # have trained on, so its baseline probe is apples-to-apples with
        # REGIME_B_TRAINED's probe on the SAME bases (rather than each arm
        # implicitly using a different basis draw). `gen` continues as the
        # SGD-sampling stream inside `_train_regime_b` when n_steps > 0.
        bases, gen = _regime_b_cluster_bases(seed, LATENT_DIM, PROBE_CLUSTERS)

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        final_h1_loss = _train_regime_b(
            agent.e1.context_memory.write_addr_tagger, bases, gen, n_steps, seed, print_every
        )

        zg.observe(agent)

        final_state = agent.e1.context_memory.write_addr_tagger.state_dict()
        max_abs_diff = 0.0
        for k, v0 in tagger_init_state.items():
            d = (final_state[k] - v0).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
        tagger_params_moved = max_abs_diff > TAGGER_MOVE_EPS

        # Finding F1 fix -- LOAD-BEARING: probe on the SAME bases actually
        # trained on, fresh noise draws from an independently-seeded
        # generator (never overlaps the continuous training stream `gen`).
        probe_jaccard_trained_clusters, probe_per_cluster_trained_clusters = (
            _run_content_probe_on_bases(agent, bases, PROBE_JITTER, PROBE_N, seed + 800_000)
        )

        # Secondary / non-gating: generalization to a wholly different,
        # unrelated draw of the underlying distribution.
        probe_jaccard_fresh_clusters, probe_per_cluster_fresh_clusters = (
            _run_content_probe(agent, seed)
        )

        row: Dict[str, Any] = {
            "regime": "B",
            "arm": arm_name,
            "seed": seed,
            "n_steps": n_steps,
            "final_h1_loss": final_h1_loss,
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_2cluster_jaccard_trained_clusters": probe_jaccard_trained_clusters,
            "probe_2cluster_occupied_trained_clusters": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster_trained_clusters.items()
            },
            "probe_2cluster_jaccard_fresh_clusters": probe_jaccard_fresh_clusters,
            "probe_2cluster_occupied_fresh_clusters": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster_fresh_clusters.items()
            },
        }
        cell.stamp(row)

    print(
        f"verdict: {'PASS' if n_steps == 0 or tagger_params_moved else 'FAIL'} "
        f"(regime=B training-engagement control; "
        f"probe_jaccard_trained_clusters={probe_jaccard_trained_clusters:.3f}, "
        f"probe_jaccard_fresh_clusters={probe_jaccard_fresh_clusters:.3f})",
        flush=True,
    )
    return row


# ------------------------------------------------------------------ #
# Top-level run                                                            #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    training_episodes = 2 if dry_run else TRAINING_EPISODES
    steps_per_episode = 5 if dry_run else STEPS_PER_EPISODE
    n_steps_b_trained = 6 if dry_run else N_STEPS_B
    print_every_b = 2 if dry_run else PRINT_EVERY_B
    seeds = SEEDS[:1] if dry_run else SEEDS

    base_config_slice: Dict[str, Any] = {
        "seeds": seeds,
        "lambda_h1": LAMBDA_H1,
        "training_episodes": training_episodes,
        "steps_per_episode": steps_per_episode,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "n_steps_b_trained": n_steps_b_trained,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "latent_dim": LATENT_DIM,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
    }

    # --- Regime A ---
    rows_a: List[Dict[str, Any]] = []
    for arm_name, lambda_h1 in ARMS_A:
        for seed in seeds:
            rows_a.append(_run_cell_a(
                arm_name, lambda_h1, seed, base_config_slice, zg,
                training_episodes, steps_per_episode,
            ))

    # --- Regime B ---
    rows_b: List[Dict[str, Any]] = []
    for arm_name, n_steps in ARMS_B:
        this_n_steps = 0 if n_steps == 0 else n_steps_b_trained
        for seed in seeds:
            rows_b.append(_run_cell_b(
                arm_name, this_n_steps, seed, base_config_slice, zg, print_every_b,
            ))

    by_arm_a: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in ARMS_A}
    for row in rows_a:
        by_arm_a[row["arm"]].append(row)
    by_arm_b: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in ARMS_B}
    for row in rows_b:
        by_arm_b[row["arm"]].append(row)

    def _mean_field(by_arm: Dict[str, List[Dict[str, Any]]], arm_name: str,
                     value_key: str) -> float:
        """Skips None entries (Finding F2: a seed whose buffer never grew
        past N_HELDOUT + MIN_H1_BUF has no held-out probe value)."""
        vals = [r[value_key] for r in by_arm[arm_name] if r.get(value_key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    min_write_calls_a = min((r["n_write_calls"] for r in rows_a), default=0.0)
    # Finding F2: Regime A's LOAD-BEARING readout is now the held-out
    # real-latent probe, not the generic synthetic 2-cluster probe.
    mean_jaccard_untrained_a = _mean_field(
        by_arm_a, "REGIME_A_UNTRAINED", "probe_heldout_real_latent_jaccard")
    mean_jaccard_trained_a = _mean_field(
        by_arm_a, "REGIME_A_TRAINED", "probe_heldout_real_latent_jaccard")
    mean_jaccard_untrained_a_synthetic = _mean_field(
        by_arm_a, "REGIME_A_UNTRAINED", "probe_2cluster_jaccard_synthetic")
    mean_jaccard_trained_a_synthetic = _mean_field(
        by_arm_a, "REGIME_A_TRAINED", "probe_2cluster_jaccard_synthetic")
    # Finding F1: Regime B's LOAD-BEARING readout is now the probe on the
    # SAME bases the arm actually trained on, not a fresh unrelated draw.
    mean_jaccard_untrained_b = _mean_field(
        by_arm_b, "REGIME_B_UNTRAINED", "probe_2cluster_jaccard_trained_clusters")
    mean_jaccard_trained_b = _mean_field(
        by_arm_b, "REGIME_B_TRAINED", "probe_2cluster_jaccard_trained_clusters")
    mean_jaccard_untrained_b_fresh = _mean_field(
        by_arm_b, "REGIME_B_UNTRAINED", "probe_2cluster_jaccard_fresh_clusters")
    mean_jaccard_trained_b_fresh = _mean_field(
        by_arm_b, "REGIME_B_TRAINED", "probe_2cluster_jaccard_fresh_clusters")

    # Finding F4 -- paired (per-seed) permutation test, one per regime.
    diffs_a = _paired_diffs(
        by_arm_a["REGIME_A_UNTRAINED"], by_arm_a["REGIME_A_TRAINED"],
        "probe_heldout_real_latent_jaccard",
    )
    diffs_b = _paired_diffs(
        by_arm_b["REGIME_B_UNTRAINED"], by_arm_b["REGIME_B_TRAINED"],
        "probe_2cluster_jaccard_trained_clusters",
    )
    p_value_a = _permutation_test_pvalue(diffs_a) if diffs_a else float("nan")
    p_value_b = _permutation_test_pvalue(diffs_b) if diffs_b else float("nan")

    # --- Regime-conditioned precondition gate (see module docstring) ---
    SPECS = [
        PreconditionSpec(
            name="writepath_engaged",
            description="min n_write_calls across all Regime A cells",
            control="ContextMemory.write() genuinely fired during real training",
            threshold=WRITE_CALLS_FLOOR,
            direction="lower",
            applies_to=lambda ctx: ctx["regime"] == "A",
            applies_note=(
                "Regime B trains write_addr_tagger directly via SGD on a "
                "synthetic stream, with no environment and no "
                "ContextMemory.write() calls during training -- this "
                "precondition is not meaningful for it. Regime B's own "
                "readiness is by construction (N_STEPS is a fixed constant), "
                "not empirically gated."
            ),
        ),
        PreconditionSpec(
            name="untrained_baseline_headroom",
            description="mean_jaccard(<regime>_UNTRAINED) has enough headroom "
                         "below 1.0 for a 0.25-margin drop to be satisfiable",
            control="C2-style readiness -- an already-near-floor untrained "
                    "baseline would make 'materially lower' structurally "
                    "unsatisfiable regardless of training",
            threshold=JACCARD_MARGIN,
            direction="lower",
        ),
    ]

    gate_a = evaluate_arm_gate(
        "Regime_A", {"regime": "A"}, SPECS,
        measured={
            "writepath_engaged": float(min_write_calls_a),
            "untrained_baseline_headroom": mean_jaccard_untrained_a,
        },
    )
    gate_b = evaluate_arm_gate(
        "Regime_B", {"regime": "B"}, SPECS,
        measured={"untrained_baseline_headroom": mean_jaccard_untrained_b},
    )
    gate = aggregate_arm_gates([gate_a, gate_b])

    regime_a_directional_pass = mean_jaccard_trained_a <= (mean_jaccard_untrained_a - JACCARD_MARGIN)
    regime_b_directional_pass = mean_jaccard_trained_b <= (mean_jaccard_untrained_b - JACCARD_MARGIN)

    # Secondary / non-gating (Finding F4): the OLD bare-margin reading --
    # gate green AND the directional drop clears JACCARD_MARGIN, with no
    # significance test. Kept for comparison; no longer load-bearing.
    regime_a_bare_margin_pass = gate_a["gate_green"] and regime_a_directional_pass
    regime_b_bare_margin_pass = gate_b["gate_green"] and regime_b_directional_pass
    overall_bare_margin_pass = regime_a_bare_margin_pass or regime_b_bare_margin_pass

    # Finding F4 fix -- LOAD-BEARING: a bare-margin pass ALSO requires the
    # paired sign-flip permutation-test p-value to clear the
    # Bonferroni-corrected alpha (0.05 / 2 regimes). A per-seed Jaccard is
    # near-binary, so the bare mean-margin test alone has an estimated ~15%
    # false-positive rate under the null with no multiplicity correction.
    regime_a_h1_pass = (
        regime_a_bare_margin_pass
        and not math.isnan(p_value_a) and p_value_a < ALPHA_CORRECTED
    )
    regime_b_h1_pass = (
        regime_b_bare_margin_pass
        and not math.isnan(p_value_b) and p_value_b < ALPHA_CORRECTED
    )

    overall_pass = regime_a_h1_pass or regime_b_h1_pass
    status = "PASS" if overall_pass else "FAIL"

    if overall_pass and regime_a_h1_pass and regime_b_h1_pass:
        label = "h1_confirmed_both_regimes"
    elif overall_pass and regime_a_h1_pass:
        label = "h1_confirmed_real_agent_regime_only"
    elif overall_pass and regime_b_h1_pass:
        label = "h1_confirmed_synthetic_regime_only_points_to_h4_input_distribution"
    elif not gate["non_degenerate"]:
        label = "h1_both_regimes_gate_not_ready"
    else:
        label = "h1_content_referencing_objective_not_confirmed_either_regime"

    # Finding F3 fix: soften the H4-attribution reading of that specific
    # label with the residual update-budget-asymmetry caveat (H1_INJECT_
    # EVERY_N_STEPS shrank Regime A's own budget from ~95 to ~1500 updates,
    # a ~16x reduction in the asymmetry vs Regime B's 15000 -- but a ~10x
    # gap remains, so a synthetic-only pass is suggestive of H4, not
    # decisive proof of it).
    h4_attribution_caveat = (
        "(residual ~10x update-budget asymmetry between regimes; treat "
        "this reading as suggestive, not decisive, for H4 attribution "
        "specifically)"
        if label == "h1_confirmed_synthetic_regime_only_points_to_h4_input_distribution"
        else None
    )

    criteria_non_degenerate = {
        "H1_content_referencing_objective_succeeds_in_either_regime": gate["non_degenerate"],
        "H1_regime_A_content_referencing_objective_succeeds": gate_a["gate_green"],
        "H1_regime_B_content_referencing_objective_succeeds": gate_b["gate_green"],
    }

    n_tagger_moved_a_trained = sum(
        1 for r in by_arm_a["REGIME_A_TRAINED"] if r["tagger_params_moved"] is True
    )
    n_tagger_frozen_a_untrained = sum(
        1 for r in by_arm_a["REGIME_A_UNTRAINED"] if r["tagger_params_moved"] is False
    )
    n_tagger_moved_b_trained = sum(
        1 for r in by_arm_b["REGIME_B_TRAINED"] if r["tagger_params_moved"] is True
    )
    n_tagger_frozen_b_untrained = sum(
        1 for r in by_arm_b["REGIME_B_UNTRAINED"] if r["tagger_params_moved"] is False
    )

    criteria = [
        {
            "name": "H1_content_referencing_objective_succeeds_in_either_regime",
            "load_bearing": True,
            "passed": overall_pass,
            "regime_a_pass": regime_a_h1_pass,
            "regime_b_pass": regime_b_h1_pass,
            "mean_jaccard_untrained_regime_a": mean_jaccard_untrained_a,
            "mean_jaccard_trained_regime_a": mean_jaccard_trained_a,
            "mean_jaccard_untrained_regime_b": mean_jaccard_untrained_b,
            "mean_jaccard_trained_regime_b": mean_jaccard_trained_b,
            "required_margin": JACCARD_MARGIN,
            "p_value_regime_a": p_value_a,
            "p_value_regime_b": p_value_b,
            "alpha_corrected": ALPHA_CORRECTED,
            "n_paired_seeds_regime_a": len(diffs_a),
            "n_paired_seeds_regime_b": len(diffs_b),
            "combination_rule": (
                "OR across regimes -- either regime succeeding is sufficient "
                "evidence FOR H1; the null (H1 false) requires BOTH regimes "
                "to fail their own headroom-gated comparison. Per Finding "
                "F4, a regime's own PASS additionally requires its paired "
                "sign-flip permutation-test p-value to clear the "
                "Bonferroni-corrected alpha (0.05 / 2 regimes) -- a bare "
                "margin drop alone is no longer sufficient."
            ),
            "note": (
                "This is the genuinely open question. A FAIL in one regime "
                "with a PASS in the other is NOT 'mostly failed' -- see "
                "combination_rule. A FAIL in both is a real, expected-"
                "possible scientific outcome (see module docstring), not a "
                "bug."
                + (f" {h4_attribution_caveat}" if h4_attribution_caveat else "")
            ),
        },
        {
            "name": "H1_bare_margin_reading_non_gating",
            "load_bearing": False,
            "kind": "descriptive",
            "passed": overall_bare_margin_pass,
            "regime_a_pass": regime_a_bare_margin_pass,
            "regime_b_pass": regime_b_bare_margin_pass,
            "note": (
                "SECONDARY reading, kept for comparison only (Finding F4): "
                "the OLD gate-green-plus-margin-drop criterion, with no "
                "significance test. The load-bearing criterion above is "
                "the permutation-test-gated version; this field exists so "
                "a reader can see whether the statistical correction "
                "changed the verdict for either regime."
            ),
        },
        {
            "name": "P0_writepath_engaged_regime_a",
            "load_bearing": False,
            "kind": "readiness",
            "passed": gate_a["gate_green"] or (min_write_calls_a >= WRITE_CALLS_FLOOR),
            "measured": min_write_calls_a,
            "threshold": WRITE_CALLS_FLOOR,
            "direction": "lower",
        },
        {
            "name": "untrained_baseline_headroom_regime_a",
            "load_bearing": False,
            "kind": "readiness",
            "passed": mean_jaccard_untrained_a >= JACCARD_MARGIN,
            "measured": mean_jaccard_untrained_a,
            "threshold": JACCARD_MARGIN,
            "direction": "lower",
        },
        {
            "name": "untrained_baseline_headroom_regime_b",
            "load_bearing": False,
            "kind": "readiness",
            "passed": mean_jaccard_untrained_b >= JACCARD_MARGIN,
            "measured": mean_jaccard_untrained_b,
            "threshold": JACCARD_MARGIN,
            "direction": "lower",
        },
        {
            "name": "regime_a_tagger_gradient_controls",
            "load_bearing": False,
            "kind": "control",
            "passed": (
                n_tagger_frozen_a_untrained == len(by_arm_a["REGIME_A_UNTRAINED"])
                and n_tagger_moved_a_trained == len(by_arm_a["REGIME_A_TRAINED"])
            ),
            "n_untrained_frozen": n_tagger_frozen_a_untrained,
            "n_untrained_total": len(by_arm_a["REGIME_A_UNTRAINED"]),
            "n_trained_moved": n_tagger_moved_a_trained,
            "n_trained_total": len(by_arm_a["REGIME_A_TRAINED"]),
            "note": "Mechanical positive/negative control: lambda_h1=0.0 should "
                    "leave write_addr_tagger byte-identical to init; "
                    "lambda_h1=0.5 should move it.",
        },
        {
            "name": "regime_b_tagger_gradient_controls",
            "load_bearing": False,
            "kind": "control",
            "passed": (
                n_tagger_frozen_b_untrained == len(by_arm_b["REGIME_B_UNTRAINED"])
                and n_tagger_moved_b_trained == len(by_arm_b["REGIME_B_TRAINED"])
            ),
            "n_untrained_frozen": n_tagger_frozen_b_untrained,
            "n_untrained_total": len(by_arm_b["REGIME_B_UNTRAINED"]),
            "n_trained_moved": n_tagger_moved_b_trained,
            "n_trained_total": len(by_arm_b["REGIME_B_TRAINED"]),
            "note": "Mechanical positive/negative control: 0 SGD steps should "
                    "leave write_addr_tagger byte-identical to init; "
                    f"{N_STEPS_B} steps should move it.",
        },
    ]

    metrics: Dict[str, Any] = {
        "n_write_calls_per_seed_regime_a": {
            arm: [r["n_write_calls"] for r in by_arm_a[arm]] for arm, _ in ARMS_A
        },
        # Load-bearing (Finding F2): held-out real safe/dangerous-latent probe.
        "probe_heldout_real_latent_jaccard_per_seed_regime_a": {
            arm: [r["probe_heldout_real_latent_jaccard"] for r in by_arm_a[arm]] for arm, _ in ARMS_A
        },
        # Secondary / non-gating: generic synthetic 2-cluster probe.
        "probe_2cluster_jaccard_synthetic_per_seed_regime_a": {
            arm: [r["probe_2cluster_jaccard_synthetic"] for r in by_arm_a[arm]] for arm, _ in ARMS_A
        },
        # Load-bearing (Finding F1): probe on the SAME bases actually trained on.
        "probe_2cluster_jaccard_trained_clusters_per_seed_regime_b": {
            arm: [r["probe_2cluster_jaccard_trained_clusters"] for r in by_arm_b[arm]] for arm, _ in ARMS_B
        },
        # Secondary / non-gating: fresh, unrelated cluster-basis draw.
        "probe_2cluster_jaccard_fresh_clusters_per_seed_regime_b": {
            arm: [r["probe_2cluster_jaccard_fresh_clusters"] for r in by_arm_b[arm]] for arm, _ in ARMS_B
        },
        "mean_probe_heldout_real_latent_jaccard_regime_a": {
            "REGIME_A_UNTRAINED": mean_jaccard_untrained_a,
            "REGIME_A_TRAINED": mean_jaccard_trained_a,
        },
        "mean_probe_2cluster_jaccard_synthetic_regime_a": {
            "REGIME_A_UNTRAINED": mean_jaccard_untrained_a_synthetic,
            "REGIME_A_TRAINED": mean_jaccard_trained_a_synthetic,
        },
        "mean_probe_2cluster_jaccard_trained_clusters_regime_b": {
            "REGIME_B_UNTRAINED": mean_jaccard_untrained_b,
            "REGIME_B_TRAINED": mean_jaccard_trained_b,
        },
        "mean_probe_2cluster_jaccard_fresh_clusters_regime_b": {
            "REGIME_B_UNTRAINED": mean_jaccard_untrained_b_fresh,
            "REGIME_B_TRAINED": mean_jaccard_trained_b_fresh,
        },
        "permutation_test": {
            "p_value_regime_a": p_value_a,
            "p_value_regime_b": p_value_b,
            "alpha_corrected": ALPHA_CORRECTED,
            "n_paired_seeds_regime_a": len(diffs_a),
            "n_paired_seeds_regime_b": len(diffs_b),
        },
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory write-content H1 (loss-objective-mismatch) "
        f"contrastive-loss probe, dual regime\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; hypothesis_space "
        f"contextmemory_write_content_discrimination, H1-loss-objective-mismatch)\n\n"
        f"| Regime | mean Jaccard UNTRAINED | mean Jaccard TRAINED | "
        f"drop >= {JACCARD_MARGIN}? | p-value | regime gate green | "
        f"regime H1 pass (perm-test-gated) | bare-margin pass (secondary) |\n"
        f"|---|---|---|---|---|---|---|---|\n"
        f"| A (real-agent, held-out latents) | {mean_jaccard_untrained_a:.3f} | "
        f"{mean_jaccard_trained_a:.3f} | {regime_a_directional_pass} | "
        f"{p_value_a:.4f} | {gate_a['gate_green']} | {regime_a_h1_pass} | "
        f"{regime_a_bare_margin_pass} |\n"
        f"| B (synthetic, trained clusters) | {mean_jaccard_untrained_b:.3f} | "
        f"{mean_jaccard_trained_b:.3f} | {regime_b_directional_pass} | "
        f"{p_value_b:.4f} | {gate_b['gate_green']} | {regime_b_h1_pass} | "
        f"{regime_b_bare_margin_pass} |\n\n"
        f"H1_content_referencing_objective_succeeds_in_either_regime "
        f"(OR across regimes, permutation-test-gated): "
        f"{'PASS' if overall_pass else 'FAIL'} "
        f"(bare-margin-only secondary reading: "
        f"{'PASS' if overall_bare_margin_pass else 'FAIL'})\n"
        f"Bonferroni-corrected alpha: {ALPHA_CORRECTED}\n"
        f"{gate['degeneracy_reason']}\n"
        + (f"\nH4-attribution caveat: {h4_attribution_caveat}\n" if h4_attribution_caveat else "")
    )

    result: Dict[str, Any] = {
        "outcome": status,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "metrics": metrics,
        "arm_results": rows_a + rows_b,
        "summary_markdown": summary_markdown,
        "per_arm_gate": gate["per_arm_gate"],
        "non_degenerate": gate["non_degenerate"],
        "degeneracy_reason": gate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": gate["adjudication_preconditions"],
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
        },
        "fatal_error_count": 0,
    }
    return result, zg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    result, zg_accumulator = run(dry_run=args.dry_run)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH

    full_config = {
        "arms_a": [name for name, _ in ARMS_A],
        "arms_b": [name for name, _ in ARMS_B],
        "seeds": SEEDS,
        "lambda_h1": LAMBDA_H1,
        "training_episodes": TRAINING_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "n_steps_b": N_STEPS_B,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "write_calls_floor": WRITE_CALLS_FLOOR,
        "jaccard_margin": JACCARD_MARGIN,
        "min_h1_buf": MIN_H1_BUF,
        "h1_batch_per_side": H1_BATCH_PER_SIDE,
        "h1_inject_every_n_steps": H1_INJECT_EVERY_N_STEPS,  # Finding F3
        "n_heldout": N_HELDOUT,  # Finding F2
        "alpha_corrected": ALPHA_CORRECTED,  # Finding F4
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
        "latent_dim": LATENT_DIM,
    }

    out_path = write_flat_manifest(
        result,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=__file__,
        started_at=t0,
        z_goal_stream_stats=zg_accumulator.stats(),
    )

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)

    emit_outcome(
        outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
