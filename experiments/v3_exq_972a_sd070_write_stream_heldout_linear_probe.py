#!/opt/local/bin/python3
"""
V3-EXQ-972a -- SD-070: held-out LINEAR PROBE of train-time ContextMemory
write-stream separability (V3-EXQ-972 successor). Routes to SD-070.

Chip: chip-20260906-sd070-write-stream-heldout-probe
Campaign: chip-20260907-campaign-w4-s3-contextmemory-content (S3 item 2)

experiment_purpose: diagnostic
claim_ids: ["SD-070"]  (primary routing per the substrate entry's own hint (2);
                        NOT the ContextMemory entry)
sleep_driver_pattern: N/A (no sleep loop)

red-team (fable), 2026-09-07: see the RED-TEAM section at the end of this
docstring.

WHY THIS RUN EXISTS. V3-EXQ-972 (H4, input-distribution) measured the real
train-time write stream ([z_self, z_world], the exact tensor
ContextMemory.write() receives) at separability 0.028 (intra-context cosine
0.980 / 0.971 vs inter 0.948) and read it as "representation_undifferentiated".
The 2026-09-03 cluster autopsy (section 6, residual risk 2) and the substrate
entry `contextmemory-write-path-addressing-degeneracy`'s implementation_hint
(2) both say the same thing about that number: an UNCENTRED cosine cannot
distinguish "no structure" from "structure off the raw axis" -- a tight cone
with a small, consistent, linearly decodable offset between contexts reads
0.03 on cosine and 1.0 on a linear probe. The user's 2026-09-06 decision
(decision_2026_09_06) minted this run to settle it with a held-out linear
probe, and routed the answer to SD-070 because the entry's own hint names the
z_world encoder recipe as the lever for the content half:

  probe AT CHANCE on the lineage's write stream  -> the content half is
      SD-070's problem: no write-side objective can condition on structure
      that is not there (write-side objectives are moot until the encoder
      recipe changes what the stream carries).
  probe WELL ABOVE CHANCE                          -> structure exists off the
      raw axis; the ContextMemory instrument-redesign run (V3-EXQ-970a,
      chip-20260906-ctxmem-instrument-redesign-970a) is the next step, and
      a write-side objective CAN in principle condition on it.

SIBLING RESULT LANDED WHILE THIS DRIVER WAS BEING AUTHORED, and it partly
answers the "moot" branch above -- recorded here so this run's framing is not
stale on arrival, WITHOUT changing anything pre-registered below (every
criterion, threshold and arm in this file was fixed before the sibling
returned). V3-EXQ-970a PASSED on ree-cloud-2, 2026-09-07T15:22Z, all four
units gate-green with no tied pairs: on the REAL AGENT its contrastive
write-addressing objective lifted held-out 2-context routing NMI_excess from
0.018 (seed-matched untrained) to 0.312, 8/8 seeds positive, p = 0.0039,
while the substrate's content-BLIND diversity loss stayed flat (0.011,
p = 0.51). So a write-side objective is demonstrably NOT moot on this stream,
and the "probe at chance -> content half is SD-070's" branch is much less
likely a priori than it was when this chip was minted.

WHAT THAT LEAVES THIS RUN ANSWERING -- three things 970a cannot:
  (a) WHICH HALF carries the structure. 970a's write_addr_tagger reads the
      full 64-d stream, so its success cannot separate z_world (SD-070's
      lever, and H4's subject) from z_self (interoception: health falls to
      ~0 within the 3-4 steps a dangerous episode lasts). This driver's
      LOAD-BEARING readout is the zworld32 probe precisely for that reason
      (red-team F1), with zself32 and full64 reported beside it.
  (b) Whether SD-070's ENCODER RECIPE raises it (T3) -- 970a does not
      exercise SD-070 at all, and the substrate entry's own
      depends_on_unresolved names the SD-070 recipe as the content half's
      upstream gate.
  (c) LINEAR decodability. 970a's tagger is a supervised nonlinear MLP
      trained on the class labels; "an MLP can learn to route these" and
      "the classes are linearly separable in the stream" are different
      claims, and the second is what an addressing rule that must generalise
      off a linear score can actually exploit.
A T1 pass here therefore CORROBORATES and LOCALISES 970a rather than
duplicating it; a T1 null with 970a's positive would be the informative
dissociation (structure present but not linearly available in z_world).

THREE ENCODER REGIMES (arms), one cell per (arm, seed), RNG fully reset at
each cell via arm_cell:
  UNTRAINED_ENCODER  the agent at initialisation, no training of any kind;
                     the write stream is a random projection of the
                     observations. Seed-matched baseline: how much context
                     decodability the OBSERVATION hands a random encoder
                     for free (8 hazards vs 1 is visible in world_obs).
                     NOTE (a consequence, not a defect): since every arm now
                     encodes the SAME observation sequence (see COLLECTION
                     below) and LINEAGE's encoder never moves, this arm's
                     latents are BIT-IDENTICAL to LINEAGE's. That identity is
                     asserted as a positive control on both the shared-
                     observation machinery and the frozen-encoder finding, and
                     it makes T4 a mechanical identity check rather than an
                     independent baseline -- it is nonetheless left inside the
                     Bonferroni family (the conservative choice).
  LINEAGE            V3-EXQ-956/970/972's harness verbatim: gumbel_learned,
                     contextmemory_write_addressing_loss_weight = 0.5, E1/E2
                     trained online every step for TRAINING_EPISODES = 100
                     (150 steps, safe/dangerous contexts alternating every
                     CONTEXT_SWITCH_EVERY = 5 episodes). This IS the stream
                     972 measured and the stream V3-EXQ-970a's tagger sees.
                     [LOAD-BEARING arm]
                     AUTHORING-TIME FINDING (2026-09-07): this harness sends
                     NO gradient into latent_stack -- E1/E2 train on detached
                     buffered latents; measured directly: 0 of 49 latent_stack
                     parameters receive grad from the E1+E2 losses, and the
                     dry-run's zworld_encoder_guard delta is 0 of 49 tensors
                     after the online phase. The lineage's write stream is a
                     FROZEN RANDOM PROJECTION of the observation (the exact
                     V3-EXQ-780 signature the guard was built for). That is
                     not a defect of this driver -- it is the object under
                     test, recorded as the `lineage_encoder_frozen_by_
                     construction` finding for governance -- and it is why
                     the encoder-trained gate is scoped to SD070_WARMED only.
  SD070_WARMED       the SD-070 P0a z_world encoder warmup
                     (experiments/_lib/zworld_p0_warmup.run_zworld_p0,
                     ZWORLD_P0_EPISODES = 60 on a dedicated warmup env with
                     RandomPolicy -- the exact integration V3-EXQ-783
                     validated and the 1006 lineage uses; RNG-neutral by
                     construction) FOLLOWED BY the identical LINEAGE online
                     phase. The zworld_encoder_guard weight-delta check is
                     recorded after P0a (strict=False; a zero delta reds
                     the arm's gate rather than aborting the run). This is
                     "the lineage plus the recipe" -- the routing question
                     is whether the recipe changes what the stream carries
                     UNDER THIS HARNESS, so the online phase is kept, not
                     frozen out -- and since that phase cannot move the
                     encoder (see LINEAGE), this arm is exactly "SD-070-
                     trained encoder, then the lineage's own downstream
                     training", clean. [routing arm]

STEP-MATCHED CONTEXTS (the V3-EXQ-970a red-team F5 finding applies here
verbatim): in this harness dangerous episodes end in ~3.8 steps while safe
ones run ~27, and z_self is EMA-attenuated (alpha_self = 0.3) to 0.30 / 0.51 /
0.66 / 0.76 of steady state at steps 1-4 after agent.reset(). A probe on
unmatched states would decode TIME-SINCE-RESET, not context. So EVERY state
this driver records -- training-phase and held-out alike -- is from within-
episode step index < STEP_MATCH_MAX = 4, in BOTH classes; the per-class
z_self / z_world norms are recorded as the audit, and the probe is run on
the FULL write stream (64-d) and on each half separately (z_world 32-d,
z_self 32-d), so a reader can see which half carries the decodable structure.

HELD-OUT DESIGN (episode-level split; the 972 red-team's own recorded
caveat was that intra-context pairs shared episodes/blocks and no null could
be computed post hoc). After the arm's training (none, for UNTRAINED_ENCODER)
the agent runs COLLECT_EPISODES = 60 collection episodes in eval mode with NO
optimiser step -- contexts alternating in blocks of CONTEXT_SWITCH_EVERY --
and every step-matched state is recorded with its (episode, step, context).
The probe is TRAINED on the states of the first PROBE_TRAIN_FRAC = 60% of
collection episodes (blocks 0-7: 4 safe + 4 dangerous) and TESTED on the
states of the remaining 40% (blocks 8-11: 2 + 2). No state, and no episode,
is on both sides. Each side is balanced to min(n_safe, n_dangerous) by a
per-cell probe generator. A cell whose balanced TEST count per class is
< MIN_TEST_PER_CLASS = 24 contributes no load-bearing value (recorded, seed
dropped); the arm's precondition `heldout_adequate_seeds` requires
>= MIN_ADEQUATE_SEEDS = 7 adequate seeds so the exact sign-flip p-floor
(1/2**7 = 0.0078) clears alpha.

THE PROBE. L2-regularised logistic regression (one linear layer, LBFGS,
PROBE_L2 = 1e-2), features standardised with the TRAIN side's mean/sd only.
DV per (cell, feature set) = balanced test accuracy MINUS the mean balanced
test accuracy of N_PROBE_SHUFFLE = 50 refits with the TRAIN labels permuted
(the label-shuffle null the chip mandates; evaluated on the true test labels,
so it is the null of THIS probe at THIS n). NMI-style bias removal for a
classifier: the null mean of a random-feature probe is ~0.49 at this n, and
its spread is what the margin is scaled on.

AUTHORING-TIME NULL BAND (2026-09-07, random 64-d features, 150 draws):
  n_train/class 60, n_test/class 38: null bal_acc 0.491 +- 0.055 (p95 0.579, max 0.605)
  n_train/class 100, n_test/class 60: 0.500 +- 0.047 (p95 0.583)
  n_train/class 40, n_test/class 24: 0.482 +- 0.066 (p95 0.604, max 0.708)
Expected run-time sizes (after the F4 per-step stratification and the F6
block split): 60 collection episodes = 12 blocks -> 8 train blocks (20 safe +
20 dangerous episodes) and 4 test blocks (10 + 10). Safe episodes reach all
four step strata; dangerous ones reach step 4 in roughly 4 of 10, so the
stratified per-class totals are ~68 train and ~34 test. PROBE_MARGIN = 0.15
is ~2.7 null SDs at that test size and exceeds the largest null draw measured
at the smaller n_test/class = 24 row above. dv_headroom is DECLARED (kind dv_headroom, _metrics.dv_headroom_check,
statistic ceiling_headroom against the DV's analytic ceiling 1.0 above each
arm's own realised null means).

STATISTICS. Per arm A and feature set F: per-seed excess e_i; pass(A, F) iff
the arm's gate is green AND mean(e) >= PROBE_MARGIN AND the exact one-sided
sign-flip permutation p-value over seeds < ALPHA_CORRECTED = 0.05 / 4 (the
four pre-registered tests below; SEEDS has 8 entries, p-floor 1/256).
  T1 (LOAD-BEARING)  LINEAGE, full64 -- 972's question, held out.
  T2                 SD070_WARMED, full64.
  T3 (routing)       paired per-seed diff e(SD070_WARMED) - e(LINEAGE), full64
                     -- does the recipe RAISE decodability under this harness.
  T4                 UNTRAINED_ENCODER, full64 -- structure the observation
                     hands a random projection.
z_world-only and z_self-only probes are recorded for every arm (descriptive,
no test), as are the 972-style uncentred cosine separability on the SAME
step-matched states (read-across to 972's 0.028) and the z_world
participation ratio over the collected states (SD-070's own anti-collapse
DV; PR retention was the recipe's validated readout).

VERDICT GRID (interpretation.label):
  LINEAGE gate red AND SD070_WARMED gate red        -> probe_gate_not_ready
  T1 pass                                             -> lineage_stream_linearly_decodable_structure_off_raw_axis
                                                         (z_world carries linearly available context; corroborates and
                                                          LOCALISES V3-EXQ-970a's positive to the encoder's world half)
  T1 fail, T2 pass AND T3 pass                        -> lineage_stream_at_chance_sd070_recipe_restores_decodability
                                                         (z_world carries nothing linearly available until the recipe runs;
                                                          read WITH 970a: the structure its tagger exploited is then either
                                                          non-linear or in the z_self half -- check T1b / zself32)
  T1 fail, T2 pass, T3 not pass                       -> lineage_stream_at_chance_warmed_decodable_contrast_underpowered
  T1 fail, T2 fail (both gates green)                 -> write_stream_at_chance_under_both_regimes
                                                         (upstream of the encoder recipe; T4 says whether the OBSERVATION carries it)
  T1 not scorable, SD070_WARMED scorable              -> lineage_unscored_<T2 reading>
evidence_direction (SD-070): "supports" iff T3 passes (the recipe measurably
raises usable z_world structure in the write stream -- SD-070's premise),
else "non_contributory" (a null on THIS readout does not refute SD-070's own
anti-collapse / grounding claim, which V3-EXQ-783 adjudicated). Diagnostic
purpose: excluded from confidence scoring regardless; the direction is
recorded so governance's routing is honest.

Step 2.4 GOV-REUSE-1 (re-checked 2026-09-07 AFTER the 970a result landed):
the decisive readout (held-out linear-probe balanced accuracy over a
label-shuffle null, per feature half, on the write-stream latents) is
recorded in NO manifest. V3-EXQ-972 carries only scalar cosine statistics --
the per-step latents were never banked (checked
evidence/experiments/v3_exq_972_.../runs/*/manifest.json: intra/inter cosines
and separability per seed only). V3-EXQ-970a's manifest is the closest
existing evidence and is explicitly NOT a substitute: its DV is a trained
tagger's slot-routing NMI over the FULL 64-d stream, so it carries neither
the per-half localisation (a) nor any SD-070 arm (b) nor a linear readout
(c); and it records no latents, so nothing here is derivable post hoc from
it. Partially-recoverable route considered and rejected on those three
grounds. Not recoverable; run fresh.
Step 2.5 / 2.5a: SD-070 is IMPLEMENTED (ree_core/latent/zworld_p0.py,
validated V3-EXQ-783); `run_zworld_p0` + `assert_world_encoder_trained` are
the exact 1006-lineage integration; the --dry-run smoke exercises the real
trainer (batch 8 / 2 epochs) and records the encoder weight delta.
Step 2.5b re-derive brake: SD-070 count = 0 (2026-09-07). Not braked.
Step 2.5c substrate-path overlap (2026-09-07): the same four open
`corrupting` entries as 970a; contextmemory-write-path-addressing-degeneracy
(the stream's own producer, not an unrelated overlap); mode-governance-
engagement / SD-082 / SD-e1-rollout-consistency-training removed by
construction (no salience coordinator, no lateral-PFC analog, no
predict_long_horizon). No entry names ree_core/latent/zworld_p0.py.

SUBSTRATE PROPERTIES held constant (956/972 verbatim): gumbel_learned +
addressing weight 0.5 in the trained arms (the tagger is irrelevant to this
DV but keeps the stream identical to 972's); contextmemory_gated_content_
write=True; sd016_writepath_mode="sense_only"; alpha_world=0.9; use_noise_
floor=True; context_memory.memory.requires_grad_(False); per-stream VS /
anchor sets / SD-039 payload True; salience coordinator, coalition
controller, sd016_enabled at defaults (False).

ethics_preflight: all involvement flags false; decision: allow (V3
pre-ethical instrumentation; SENT-0).

RED-TEAM (fable, 2026-09-07, one pass, Step 4.5): CONTESTED -- 7 findings,
each verified against the source and FIXED (none was BLOCKING as written;
F2 was conditionally blocking and is now gated rather than hoped for).
  F1 the load-bearing full64 probe was handed the class by INTEROCEPTION --
     z_self carries `health`, which falls to ~0 within the 3-4 steps a
     dangerous episode lasts, and `harm_exposure`, which rises with it, both
     through an encoder frozen in every arm -- so a pass would have said "the
     agent knows it is dying", not "z_world carries context structure the
     cosine missed". FIX: the LOAD-BEARING feature set is now zworld32 (the
     half SD-070 and H4 are actually about) for T1/T2/T3; full64 is reported
     as T1b, descriptive, outside the Bonferroni family and outside every
     label branch, with the interoception caveat stated on the criterion.
  F2 T3's DV is bounded by 1 - acc(LINEAGE), while the dv_headroom gate
     measured headroom above the NULL (~0.5, which can never fail) -- so the
     SD-070 `supports` branch could have been structurally unreachable while
     the gate reported it satisfiable. FIX: a second dv_headroom entry
     (`dv_headroom_T3_above_lineage_accuracy`) certifies headroom above
     LINEAGE's REALISED accuracy, and T3's gate now requires it.
  F3 each arm collected under its OWN trained policy, so between-arm
     contrasts confounded the encoder with where the agent walked. FIX: the
     collection phase runs a FIXED RandomPolicy on fresh, seed-identical envs,
     so every arm's frozen encoder sees the SAME observation sequence; the
     sequence is hashed per cell and equality across arms is ASSERTED.
  F4 step-matching truncated to steps 1..4 but did not equalise the per-step
     histograms (dangerous episodes under-represent step 4), leaving ~+0.07
     of time-since-reset leakage. FIX: `_balance` now balances the classes
     PER STEP INDEX, so both classes share one step histogram.
  F5 the SD-070 P0a warmup buffered SAFE-env observations only, so a T3 null
     could not be told from "the recipe never saw a hazard". FIX:
     `_AlternatingWarmupEnv` alternates safe and dangerous warmup episodes;
     the departure from the 1006 lineage's safe-only integration is recorded
     in the manifest.
  F6 the 60% episode split straddled a context block (35 // 5 == 36 // 5).
     FIX: the split is on BLOCK boundaries (8 train blocks, 4 test).
  F7 T3 paired on the intersection of adequate seeds, so 6 pairs would put
     the sign-flip p-floor (1/64) above alpha while the gate stayed green.
     FIX: T3's gate requires >= MIN_ADEQUATE_SEEDS pairs, and `p_floor` is
     recorded beside `p_value`.
Cleared by the reviewer (not raised): the label-shuffle null is exactly
unbiased; standardisation is train-side only; the frozen-encoder finding
holds (E1/E2 train on detached buffered latents); `encoder_trained` is
correctly scoped to the SD-070 arm; the env re-randomises layout on every
reset, so the probe cannot be decoding a fixed map.
"""

import argparse
import hashlib
import itertools
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta, assert_world_encoder_trained,
)
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_972a_sd070_write_stream_heldout_linear_probe"
QUEUE_ID = "V3-EXQ-972a"
CLAIM_IDS: List[str] = ["SD-070"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS: List[int] = [42, 7, 13, 100, 200, 300, 400, 500]

ARMS: List[Dict[str, Any]] = [
    {"name": "UNTRAINED_ENCODER", "online_training": False, "sd070_warmup": False},
    {"name": "LINEAGE", "online_training": True, "sd070_warmup": False},
    {"name": "SD070_WARMED", "online_training": True, "sd070_warmup": True},
]
LOAD_BEARING_ARM = "LINEAGE"
ROUTING_ARM = "SD070_WARMED"
BASELINE_ARM = "UNTRAINED_ENCODER"

# 956/972 harness, verbatim.
WRITE_SELECTION = "gumbel_learned"
WRITE_ADDRESSING_LOSS_WEIGHT = 0.5
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 100
COLLECT_EPISODES = 60
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000
ZWORLD_P0_EPISODES = 60          # SD-070 warmup -- matches the 1006 lineage
STEP_MATCH_MAX = 4               # 970a red-team F5: steps 1..4 only, both classes

PROBE_TRAIN_FRAC = 2.0 / 3.0     # BLOCK-level split of the collection phase (red-team F6):
                                 # 60 episodes / blocks of 5 = 12 blocks -> 8 train (4+4), 4 test (2+2)
COLLECT_ENV_SEED_OFFSET = 50_000 # collection envs are fresh, seed-identical across arms (red-team F3)
COLLECT_POLICY_SEED_OFFSET = 900_000  # fixed RandomPolicy for the shared observation set (F3)
MIN_TEST_PER_CLASS = 24
MIN_ADEQUATE_SEEDS = 7
ADEQUATE_SEEDS_THRESHOLD = MIN_ADEQUATE_SEEDS - 0.5   # strict floor ">" == ">= 7"
N_PROBE_SHUFFLE = 50
PROBE_L2 = 1e-2
PROBE_LBFGS_ITERS = 200
PROBE_MARGIN = 0.15
PROBE_DV_BOUNDS = (0.0, 1.0)
N_TESTS = 4
ALPHA_CORRECTED = 0.05 / N_TESTS

NUM_SLOTS = 16
LATENT_DIM = 64
HALF = LATENT_DIM // 2
FEATURE_SETS = {"full64": (0, LATENT_DIM), "zworld32": (HALF, LATENT_DIM), "zself32": (0, HALF)}
FEATURE_SEED_OFFSET = {"full64": 1, "zworld32": 2, "zself32": 3}  # fixed: hash() is per-process
LOAD_BEARING_FEATURES = "zworld32"   # red-team F1: the z_world half is what SD-070 and H4 are about
STREAM_FEATURES = "full64"            # the stream as ContextMemory sees it -- descriptive (interoception caveat)

WRITE_CALLS_FLOOR = 200.0
SEP_SUBSAMPLE_MAX = 500


# ------------------------------------------------------------------ #
# Env / agent helpers (956/972 verbatim)                                    #
# ------------------------------------------------------------------ #

def _env_kwargs_safe() -> Dict[str, Any]:
    return dict(size=10, num_hazards=1, num_resources=4, hazard_harm=0.02,
                env_drift_interval=50, env_drift_prob=0.05, proximity_harm_scale=0.10,
                proximity_benefit_scale=0.18, proximity_approach_threshold=0.15,
                hazard_field_decay=0.5, energy_decay=0.005, use_proxy_fields=True,
                resource_respawn_on_consume=True)


def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **_env_kwargs_safe())


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000, size=10, num_hazards=8, num_resources=4, hazard_harm=0.05,
        env_drift_interval=50, env_drift_prob=0.05, proximity_harm_scale=0.15,
        proximity_benefit_scale=0.18, proximity_approach_threshold=0.15,
        hazard_field_decay=0.5, energy_decay=0.005, use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


class _AlternatingWarmupEnv:
    """SD-070 P0a warmup env that alternates SAFE and DANGEROUS episodes on reset()
    (red-team F5): a safe-only warmup buffer carries near-zero variance on the hazard
    channels that separate the two collection contexts, so a T3 null could not be
    told from "the recipe never saw a hazard". Departs deliberately from the 1006
    lineage's safe-only integration; the departure is recorded in the manifest."""

    def __init__(self, seed: int) -> None:
        self._envs = [_make_env_safe(seed), _make_env_dangerous(seed)]
        self._i = -1
        self.action_dim = self._envs[0].action_dim
        self.body_obs_dim = self._envs[0].body_obs_dim
        self.world_obs_dim = self._envs[0].world_obs_dim

    @property
    def current(self):
        return self._envs[self._i % 2]

    def reset(self):
        self._i += 1
        return self.current.reset()

    def step(self, action):
        return self.current.step(action)

    def __getattr__(self, name):
        return getattr(self._envs[max(self._i, 0) % 2], name)


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        alpha_world=ALPHA_WORLD, alpha_self=0.3, reafference_action_dim=0,
        novelty_bonus_weight=0.0, sd016_writepath_mode=SD016_WRITEPATH_MODE,
        use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True,
        contextmemory_gated_content_write=CONTEXTMEMORY_GATED_CONTENT_WRITE,
        contextmemory_write_selection=WRITE_SELECTION,
        contextmemory_write_addressing_loss_weight=WRITE_ADDRESSING_LOSS_WEIGHT,
        use_noise_floor=USE_NOISE_FLOOR, noise_floor_alpha=NOISE_FLOOR_ALPHA,
        noise_floor_min_temperature=NOISE_FLOOR_MIN_TEMPERATURE,
    )
    agent = REEAgent(cfg)
    assert agent.noise_floor is not None
    assert agent.coalition is None, "write_gate() scaling at the write() call site must be identity"
    cm = agent.e1.context_memory
    assert cm.num_slots == NUM_SLOTS and cm.write_addr_tagger is not None
    assert agent.e1.config.self_dim + agent.e1.config.world_dim == LATENT_DIM
    assert agent.e1.config.self_dim == HALF, "z_self must be the first HALF dims of the write stream"
    cm.memory.requires_grad_(False)
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
        harms = []
        for a in range(num_actions):
            a_oh = _action_onehot(a, num_actions, z_world.device)
            harms.append(agent.e3.harm_eval(agent.e2.world_forward(z_world, a_oh)).mean().item())
        probs = F.softmax(-torch.tensor(harms, dtype=torch.float32) / _effective_temperature(agent), dim=0)
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
                assert n_new == 1
                self.sequence.append(int(idx))
            self._last_total = total


# ------------------------------------------------------------------ #
# The probe                                                                  #
# ------------------------------------------------------------------ #

def _fit_logistic(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros(x.shape[1], 1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = optim.LBFGS([w, b], max_iter=PROBE_LBFGS_ITERS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = (x @ w + b).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y) + PROBE_L2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def _balanced_accuracy(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> float:
    pred = ((x @ w + b).squeeze(1) > 0).to(torch.float64)
    tpr = (pred[y == 1] == 1).to(torch.float64).mean()
    tnr = (pred[y == 0] == 0).to(torch.float64).mean()
    return float((tpr + tnr) / 2)


def _balance(states: List[torch.Tensor], labels: List[int], steps: List[int], gen: torch.Generator
             ) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Balance the two classes PER STEP INDEX (red-team F4): within each within-episode
    step stratum s in 1..STEP_MATCH_MAX the larger class is subsampled to the smaller with
    `gen`, so the two classes share the same step histogram and time-since-reset cannot
    leak class. Returns X [n, D] float64, y [n] float64 and the per-class total."""
    keep: List[int] = []
    for s_idx in range(1, STEP_MATCH_MAX + 1):
        idx0 = [i for i, (c, st) in enumerate(zip(labels, steps)) if c == 0 and st == s_idx]
        idx1 = [i for i, (c, st) in enumerate(zip(labels, steps)) if c == 1 and st == s_idx]
        n_s = min(len(idx0), len(idx1))
        if n_s == 0:
            continue
        keep += [idx0[i] for i in torch.randperm(len(idx0), generator=gen)[:n_s].tolist()]
        keep += [idx1[i] for i in torch.randperm(len(idx1), generator=gen)[:n_s].tolist()]
    if not keep:
        return torch.zeros(0, LATENT_DIM, dtype=torch.float64), torch.zeros(0, dtype=torch.float64), 0
    x = torch.cat([states[i] for i in keep], dim=0).to(torch.float64)
    y = torch.tensor([labels[i] for i in keep], dtype=torch.float64)
    return x, y, int(sum(1 for i in keep if labels[i] == 0))


def linear_probe_readout(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor,
                         y_te: torch.Tensor, feature_slice: Tuple[int, int],
                         n_shuffle: int, shuffle_seed: int) -> Dict[str, Any]:
    """Balanced test accuracy of an L2-logistic probe on `feature_slice`, minus the
    mean of `n_shuffle` refits with permuted TRAIN labels (evaluated on the true
    test labels). Standardisation uses the train side only."""
    lo, hi = feature_slice
    xtr = x_tr[:, lo:hi]
    xte = x_te[:, lo:hi]
    if xtr.shape[0] < 4 or xte.shape[0] < 2:
        return {"balanced_accuracy": float("nan"), "null_mean": float("nan"), "null_sd": float("nan"),
                "excess": float("nan"), "n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]),
                "n_shuffle": 0}
    mu = xtr.mean(0, keepdim=True)
    sd = xtr.std(0, keepdim=True).clamp(min=1e-6)
    xtr_s = (xtr - mu) / sd
    xte_s = (xte - mu) / sd
    w, b = _fit_logistic(xtr_s, y_tr)
    acc = _balanced_accuracy(xte_s, y_te, w, b)
    gen = torch.Generator().manual_seed(shuffle_seed)
    nulls: List[float] = []
    for _ in range(n_shuffle):
        y_perm = y_tr[torch.randperm(y_tr.shape[0], generator=gen)]
        w0, b0 = _fit_logistic(xtr_s, y_perm)
        nulls.append(_balanced_accuracy(xte_s, y_te, w0, b0))
    null_mean = sum(nulls) / len(nulls) if nulls else float("nan")
    null_sd = ((sum((v - null_mean) ** 2 for v in nulls) / max(len(nulls) - 1, 1)) ** 0.5
               if len(nulls) > 1 else float("nan"))
    return {"balanced_accuracy": acc, "null_mean": null_mean, "null_sd": null_sd,
            "null_max": max(nulls) if nulls else float("nan"),
            "excess": acc - null_mean if null_mean == null_mean else float("nan"),
            "n_train": int(xtr.shape[0]), "n_test": int(xte.shape[0]), "n_shuffle": len(nulls)}


def _participation_ratio(x: torch.Tensor) -> float:
    if x.shape[0] < 3:
        return float("nan")
    xc = x - x.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = float(ev.sum())
    s2 = float((ev ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else float("nan")


def _cosine_separability(x0: torch.Tensor, x1: torch.Tensor, gen: torch.Generator) -> Dict[str, Any]:
    """972's uncentred-cosine statistic on the same states (read-across only)."""
    if x0.shape[0] < 2 or x1.shape[0] < 2:
        return {"intra_safe_cosine": float("nan"), "intra_dangerous_cosine": float("nan"),
                "inter_cosine": float("nan"), "separability_score": float("nan")}
    if x0.shape[0] > SEP_SUBSAMPLE_MAX:
        x0 = x0[torch.randperm(x0.shape[0], generator=gen)[:SEP_SUBSAMPLE_MAX]]
    if x1.shape[0] > SEP_SUBSAMPLE_MAX:
        x1 = x1[torch.randperm(x1.shape[0], generator=gen)[:SEP_SUBSAMPLE_MAX]]
    a = F.normalize(x0, dim=-1)
    d = F.normalize(x1, dim=-1)
    inter = float((a @ d.T).mean())
    ma = ~torch.eye(a.shape[0], dtype=torch.bool)
    md = ~torch.eye(d.shape[0], dtype=torch.bool)
    ia = float((a @ a.T)[ma].mean())
    idg = float((d @ d.T)[md].mean())
    return {"intra_safe_cosine": ia, "intra_dangerous_cosine": idg, "inter_cosine": inter,
            "separability_score": (ia + idg) / 2.0 - inter}


# ------------------------------------------------------------------ #
# Sign-flip permutation test (970/970a verbatim)                              #
# ------------------------------------------------------------------ #

def _permutation_test_pvalue(diffs: List[float], n_perm: int = 100_000, perm_seed: int = 0) -> float:
    diffs_t = [float(d) for d in diffs]
    n = len(diffs_t)
    if n == 0:
        return float("nan")
    observed = sum(diffs_t) / n
    if n <= 20:
        extreme = total = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            if sum(s * d for s, d in zip(signs, diffs_t)) / n >= observed - 1e-12:
                extreme += 1
            total += 1
        return extreme / total
    rng = torch.Generator().manual_seed(perm_seed)
    extreme = 0
    for _ in range(n_perm):
        signs = (torch.randint(0, 2, (n,), generator=rng) * 2 - 1).tolist()
        if sum(s * d for s, d in zip(signs, diffs_t)) / n >= observed - 1e-12:
            extreme += 1
    return extreme / n_perm


def _mean(vals: Sequence[float]) -> float:
    xs = [float(v) for v in vals if v is not None and v == v]
    return sum(xs) / len(xs) if xs else float("nan")


# ------------------------------------------------------------------ #
# Episode / cell runners                                                     #
# ------------------------------------------------------------------ #

def _run_episode(agent: REEAgent, env: CausalGridWorldV2, steps: int, optimizer, harm_eval_opt,
                 harm_buf_pos, harm_buf_neg, records: List[Dict[str, Any]], ep_idx: int,
                 cls: int, tracker: _WriteSequenceTracker, train: bool,
                 policy: Any = None, obs_hasher: Any = None) -> None:
    """956/972's episode runner; step-matched write-stream capture (steps 1..STEP_MATCH_MAX)
    into `records` as {"state", "episode", "step", "cls"}; train=False = eval-mode
    collection with no optimiser step of any kind."""
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    for _step in range(steps):
        latent = agent.sense(obs_dict["body_state"], obs_dict["world_state"],
                             obs_harm=obs_dict.get("harm_obs", None))
        if _step < STEP_MATCH_MAX:
            state = torch.cat([latent.z_self.detach(), latent.z_world.detach()], dim=-1).cpu()
            records.append({"state": state, "episode": ep_idx, "step": _step + 1, "cls": cls})
            if obs_hasher is not None:
                obs_hasher.update(obs_dict["body_state"].detach().cpu().numpy().tobytes())
                obs_hasher.update(obs_dict["world_state"].detach().cpu().numpy().tobytes())
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            agent._e1_tick(latent)
        tracker.poll(agent.e1.context_memory)
        z_world = latent.z_world.detach().clone()
        if policy is not None:
            # Collection phase (red-team F3): a FIXED policy, seed-identical across arms,
            # so every arm's frozen encoder sees the SAME observation sequence and the
            # between-arm contrast is an encoder contrast, not a policy one.
            action_idx = int(policy.act(env, obs_dict))
        else:
            action_idx = _select_action_baseline(agent, z_world, env.action_dim)
        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh
        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0
        if train:
            total = agent.compute_prediction_loss() + agent.compute_e2_loss()
            if total.requires_grad:
                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
            (harm_buf_pos if is_harm else harm_buf_neg).append(z_world)
            if len(harm_buf_pos) > MAX_HARM_BUF:
                del harm_buf_pos[:-MAX_HARM_BUF]
            if len(harm_buf_neg) > MAX_HARM_BUF:
                del harm_buf_neg[:-MAX_HARM_BUF]
            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_b = torch.cat([torch.cat([harm_buf_pos[i] for i in pos_idx], 0),
                                  torch.cat([harm_buf_neg[i] for i in neg_idx], 0)], 0)
                target = torch.cat([torch.ones(k_pos, 1, device=agent.device),
                                    torch.zeros(k_neg, 1, device=agent.device)], 0)
                h_loss = F.binary_cross_entropy_with_logits(agent.e3.harm_eval_head(zw_b), target)
                harm_eval_opt.zero_grad()
                h_loss.backward()
                harm_eval_opt.step()
        if done:
            break


def _split_probe(records: List[Dict[str, Any]], train_episodes: set, test_episodes: set,
                 gen: torch.Generator) -> Dict[str, Any]:
    tr = [r for r in records if r["episode"] in train_episodes]
    te = [r for r in records if r["episode"] in test_episodes]
    x_tr, y_tr, n_tr = _balance([r["state"] for r in tr], [r["cls"] for r in tr], [r["step"] for r in tr], gen)
    x_te, y_te, n_te = _balance([r["state"] for r in te], [r["cls"] for r in te], [r["step"] for r in te], gen)
    return {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te,
            "n_train_per_class": n_tr, "n_test_per_class": n_te,
            "n_train_raw": {"safe": sum(1 for r in tr if r["cls"] == 0), "dangerous": sum(1 for r in tr if r["cls"] == 1)},
            "n_test_raw": {"safe": sum(1 for r in te if r["cls"] == 0), "dangerous": sum(1 for r in te if r["cls"] == 1)}}


def _run_cell(arm: Dict[str, Any], seed: int, base_config_slice: Dict[str, Any],
              zg: ZGoalStreamAccumulator, training_episodes: int, collect_episodes: int,
              steps_per_episode: int, context_switch_every: int, zworld_p0_episodes: int,
              n_shuffle: int, dry_run: bool) -> Dict[str, Any]:
    arm_name = arm["name"]
    cell_config_slice = {**base_config_slice, "arm": arm_name,
                         "online_training": arm["online_training"], "sd070_warmup": arm["sd070_warmup"]}
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=(arm_name != BASELINE_ARM)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe)
        print(f"Seed {seed} Condition {arm_name}", flush=True)

        # --- P0a: SD-070 warmup (SD070_WARMED only), guarded ---
        warmup_report: Dict[str, Any] = {"p0a_ran": False, "guard_checked": False}
        if arm["sd070_warmup"]:
            before = latent_stack_snapshot(agent)
            warmup_env = _AlternatingWarmupEnv(seed)  # red-team F5: safe AND dangerous contexts
            p0a = run_zworld_p0(agent, warmup_env, seed, zworld_p0_episodes, steps_per_episode,
                                policy=RandomPolicy(seed), label=f"{EXPERIMENT_TYPE} P0a",
                                dry_run=dry_run)
            guard = assert_world_encoder_trained(
                agent, before, p0=zworld_p0_episodes, strict=False, context=EXPERIMENT_TYPE,
                escape_hint="SD070_WARMED arm: a zero delta reds this arm's gate")
            warmup_report = {**p0a, **guard}

        # --- online lineage phase (LINEAGE, SD070_WARMED) ---
        train_records: List[Dict[str, Any]] = []
        tracker = _WriteSequenceTracker()
        online_delta: Dict[str, Any] = {"n_changed": 0, "checked": False}  # n_changed mirrors the guard's n_latent_stack_changed
        n_train_writes = 0
        if arm["online_training"]:
            standard_params = [p for n, p in agent.named_parameters()
                               if "harm_eval_head" not in n and "context_memory.memory" not in n]
            optimizer = optim.Adam(standard_params, lr=1e-3)
            harm_eval_opt = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
            harm_buf_pos: List[torch.Tensor] = []
            harm_buf_neg: List[torch.Tensor] = []
            before_online = latent_stack_snapshot(agent)
            agent.train()
            for ep in range(training_episodes):
                is_safe = ((ep // context_switch_every) % 2 == 0)
                _run_episode(agent, env_safe if is_safe else env_dang, steps_per_episode, optimizer,
                             harm_eval_opt, harm_buf_pos, harm_buf_neg, train_records, ep,
                             0 if is_safe else 1, tracker, True)
                if (ep + 1) % 20 == 0 or (ep + 1) == training_episodes:
                    print(f"  [train] arm={arm_name} seed={seed} ep {ep + 1}/{training_episodes + collect_episodes} "
                          f"phase=online n_writes={len(tracker.sequence)} n_records={len(train_records)}", flush=True)
            n_train_writes = len(tracker.sequence)
            d = latent_stack_weight_delta(agent, before_online)
            online_delta = {"checked": True, **{k: v for k, v in d.items() if not isinstance(v, (list, dict))},
                            "n_changed": int(d.get("n_latent_stack_changed", 0) or 0),
                            "zworld_encoder_trained": d.get("zworld_encoder_trained")}

        # --- collection phase: eval mode, no optimiser, episode-indexed records ---
        collect_records: List[Dict[str, Any]] = []
        agent.eval()
        col_env_safe = _make_env_safe(seed + COLLECT_ENV_SEED_OFFSET)
        col_env_dang = _make_env_dangerous(seed + COLLECT_ENV_SEED_OFFSET)
        col_policy = RandomPolicy(seed + COLLECT_POLICY_SEED_OFFSET)
        obs_hasher = hashlib.sha256()
        for j in range(collect_episodes):
            is_safe = ((j // context_switch_every) % 2 == 0)
            _run_episode(agent, col_env_safe if is_safe else col_env_dang, steps_per_episode, None, None,
                         [], [], collect_records, j, 0 if is_safe else 1, tracker, False,
                         policy=col_policy, obs_hasher=obs_hasher)
            ep_total = training_episodes + j + 1 if arm["online_training"] else j + 1
            ep_denom = training_episodes + collect_episodes if arm["online_training"] else collect_episodes
            if (j + 1) % 20 == 0 or (j + 1) == collect_episodes:
                print(f"  [train] arm={arm_name} seed={seed} ep {ep_total}/{ep_denom} phase=collect "
                      f"n_records={len(collect_records)}", flush=True)
        zg.observe(agent)

        # --- probes ---
        n_blocks = max(collect_episodes // context_switch_every, 1)
        n_probe_train_eps = int(round(n_blocks * PROBE_TRAIN_FRAC)) * context_switch_every  # block boundary (F6)
        train_eps = set(range(0, n_probe_train_eps))
        test_eps = set(range(n_probe_train_eps, collect_episodes))
        gen = torch.Generator().manual_seed(seed + 600_000)
        split = _split_probe(collect_records, train_eps, test_eps, gen)
        probes: Dict[str, Any] = {}
        for fname, fslice in FEATURE_SETS.items():
            probes[fname] = linear_probe_readout(split["x_tr"], split["y_tr"], split["x_te"], split["y_te"],
                                                fslice, n_shuffle, seed + 610_000 + FEATURE_SEED_OFFSET[fname])
        adequate = split["n_test_per_class"] >= MIN_TEST_PER_CLASS

        # secondary: train-phase stream, episode-level split (drift-confounded, descriptive)
        train_probe: Dict[str, Any] = {}
        if arm["online_training"] and train_records:
            n_tp = int(round(max(training_episodes // context_switch_every, 1) * PROBE_TRAIN_FRAC)) * context_switch_every
            tsplit = _split_probe(train_records, set(range(0, n_tp)), set(range(n_tp, training_episodes)),
                                  torch.Generator().manual_seed(seed + 620_000))
            train_probe = {fname: linear_probe_readout(tsplit["x_tr"], tsplit["y_tr"], tsplit["x_te"],
                                                       tsplit["y_te"], fslice, max(n_shuffle // 5, 5),
                                                       seed + 630_000 + FEATURE_SEED_OFFSET[fname])
                           for fname, fslice in FEATURE_SETS.items()}
            train_probe["n_train_per_class"] = tsplit["n_train_per_class"]
            train_probe["n_test_per_class"] = tsplit["n_test_per_class"]

        # descriptive: latent norms per class, PR of z_world, 972-style cosine read-across
        x_all = torch.cat([r["state"] for r in collect_records], 0).to(torch.float64) if collect_records else torch.zeros(0, LATENT_DIM, dtype=torch.float64)
        c_all = torch.tensor([r["cls"] for r in collect_records], dtype=torch.long) if collect_records else torch.zeros(0, dtype=torch.long)
        x0 = x_all[c_all == 0]
        x1 = x_all[c_all == 1]
        norms = {
            "safe_z_self": float(x0[:, :HALF].norm(dim=-1).mean()) if x0.shape[0] else float("nan"),
            "safe_z_world": float(x0[:, HALF:].norm(dim=-1).mean()) if x0.shape[0] else float("nan"),
            "dangerous_z_self": float(x1[:, :HALF].norm(dim=-1).mean()) if x1.shape[0] else float("nan"),
            "dangerous_z_world": float(x1[:, HALF:].norm(dim=-1).mean()) if x1.shape[0] else float("nan"),
        }
        pr = {"zworld_participation_ratio": _participation_ratio(x_all[:, HALF:]),
              "zself_participation_ratio": _participation_ratio(x_all[:, :HALF]),
              "full_participation_ratio": _participation_ratio(x_all)}
        cosine = _cosine_separability(x0, x1, torch.Generator().manual_seed(seed + 640_000))
        step_hist = {
            "collect_safe": [sum(1 for r in collect_records if r["cls"] == 0 and r["step"] == s) for s in range(1, STEP_MATCH_MAX + 1)],
            "collect_dangerous": [sum(1 for r in collect_records if r["cls"] == 1 and r["step"] == s) for s in range(1, STEP_MATCH_MAX + 1)],
        }

        latent_hasher = hashlib.sha256()
        for r in collect_records:
            latent_hasher.update(r["state"].detach().cpu().numpy().tobytes())
        lb = probes[LOAD_BEARING_FEATURES]
        stream = probes[STREAM_FEATURES]
        row: Dict[str, Any] = {
            "arm": arm_name, "seed": seed,
            "online_training": arm["online_training"], "sd070_warmup": arm["sd070_warmup"],
            "n_write_calls_training": n_train_writes,
            "n_write_calls_total_incl_collect": len(tracker.sequence),
            "sd070_warmup_report": warmup_report,
            "online_latent_stack_delta": online_delta,
            "n_collect_records": len(collect_records),
            "n_train_records": len(train_records),
            "probe_split": {k: v for k, v in split.items() if not k.startswith(("x_", "y_"))},
            "heldout_adequate": adequate,
            "collection_obs_hash": obs_hasher.hexdigest(),
            "collection_latent_hash": latent_hasher.hexdigest(),
            "n_probe_train_episodes": n_probe_train_eps,
            # LOAD-BEARING readout: z_world half (None when the split was inadequate)
            "probe_excess_zworld32": lb["excess"] if adequate else None,
            "probe_balanced_accuracy_zworld32": lb["balanced_accuracy"],
            # the stream as ContextMemory sees it (interoception caveat, red-team F1)
            "probe_excess_full64": stream["excess"] if adequate else None,
            "probe_balanced_accuracy_full64": stream["balanced_accuracy"],
            "probes": probes,
            "train_phase_probes": train_probe,
            "class_latent_norms": norms,
            "participation_ratio": pr,
            "cosine_separability_972_style": cosine,
            "step_index_histogram": step_hist,
            "step_match_max": STEP_MATCH_MAX,
        }
        cell.stamp(row)

    engaged = (not arm["online_training"]) or n_train_writes >= WRITE_CALLS_FLOOR
    print(f"verdict: {'PASS' if engaged else 'FAIL'} (arm={arm_name}; n_write_calls={n_train_writes}, "
          f"probe_zworld32 acc={lb['balanced_accuracy']:.3f} null={lb['null_mean']:.3f} excess={lb['excess']:.3f} "
          f"full64 excess={stream['excess']:.3f} "
          f"n_test/class={split['n_test_per_class']} adequate={adequate}; "
          f"zworld_PR={pr['zworld_participation_ratio']:.2f}; cos_sep={cosine['separability_score']:.4f})",
          flush=True)
    return row


# ------------------------------------------------------------------ #
# Top-level run                                                              #
# ------------------------------------------------------------------ #

def _build_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec(
            name="writepath_engaged",
            description="min training-phase n_write_calls over the arm's cells",
            control="ContextMemory.write() genuinely fired during real training (956/943/970 floor)",
            threshold=WRITE_CALLS_FLOOR, direction="lower",
            applies_to=lambda ctx: ctx["online_training"],
            applies_note="UNTRAINED_ENCODER has no training phase; its write stream is a random projection by design.",
        ),
        PreconditionSpec(
            name="encoder_trained",
            description="fraction of the arm's cells whose world encoder the SD-070 P0a warmup moved "
                        "(zworld_encoder_guard weight-delta after P0a)",
            control="experiments/_lib/zworld_encoder_guard -- a frozen random projection presenting as "
                    "a trained encoder is the V3-EXQ-780 failure; this arm's premise is a TRAINED encoder",
            threshold=0.999, direction="lower",
            applies_to=lambda ctx: ctx["sd070_warmup"],
            applies_note="LINEAGE's encoder is FROZEN BY CONSTRUCTION (the 956/970/972 harness sends no "
                         "gradient into latent_stack -- measured at authoring time, 0 of 49 parameters "
                         "receive grad from the E1/E2 losses), and that frozen stream IS the object under "
                         "test, so a trained-encoder gate is not meaningful for it; UNTRAINED_ENCODER is "
                         "the deliberate frozen baseline. Encoder movement is RECORDED for every arm "
                         "(online_latent_stack_delta) and reported as the lineage_encoder_frozen finding.",
        ),
        PreconditionSpec(
            name="heldout_adequate_seeds",
            description=f"number of the arm's seeds with a balanced held-out TEST count >= {MIN_TEST_PER_CLASS} "
                        f"per class (>= {MIN_ADEQUATE_SEEDS}; strict floor {ADEQUATE_SEEDS_THRESHOLD})",
            control="episode-level split of an eval-mode collection phase; an inadequate seed is dropped, "
                    "never fabricated; enough seeds keep the sign-flip p-floor under alpha",
            threshold=ADEQUATE_SEEDS_THRESHOLD, direction="lower",
        ),
        PreconditionSpec(
            name="dv_headroom_probe_excess",
            description="ceiling headroom of the DV (balanced accuracy) above the arm's own realised "
                        "shuffle-null means: 1.0 - max_seed(null_mean) must exceed PROBE_MARGIN "
                        "(experiments/_metrics.dv_headroom_check, ceiling_headroom)",
            control="the arm's own label-shuffle null at run-time n",
            threshold=PROBE_MARGIN, direction="lower", kind="dv_headroom",
        ),
    ]


def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()
    training_episodes = 6 if dry_run else TRAINING_EPISODES
    collect_episodes = 10 if dry_run else COLLECT_EPISODES
    steps_per_episode = 15 if dry_run else STEPS_PER_EPISODE
    context_switch_every = 1 if dry_run else CONTEXT_SWITCH_EVERY
    zworld_p0_episodes = 3 if dry_run else ZWORLD_P0_EPISODES
    n_shuffle = 8 if dry_run else N_PROBE_SHUFFLE
    seeds = SEEDS[:1] if dry_run else SEEDS

    base_config_slice: Dict[str, Any] = {
        "seeds": seeds, "training_episodes": training_episodes, "collect_episodes": collect_episodes,
        "steps_per_episode": steps_per_episode, "context_switch_every": context_switch_every,
        "zworld_p0_episodes": zworld_p0_episodes, "step_match_max": STEP_MATCH_MAX,
        "write_selection": WRITE_SELECTION, "write_addressing_loss_weight": WRITE_ADDRESSING_LOSS_WEIGHT,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE, "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR, "num_slots": NUM_SLOTS, "latent_dim": LATENT_DIM,
        "probe_train_frac": PROBE_TRAIN_FRAC, "probe_l2": PROBE_L2, "n_probe_shuffle": n_shuffle,
    }

    specs = _build_specs()
    units = [{"id": a["name"], "arm": a["name"], "online_training": a["online_training"],
              "sd070_warmup": a["sd070_warmup"]} for a in ARMS]
    gate_audit = assert_no_structurally_unsatisfiable_gate(specs, units)

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_cell(arm, seed, base_config_slice, zg, training_episodes, collect_episodes,
                                  steps_per_episode, context_switch_every, zworld_p0_episodes, n_shuffle,
                                  dry_run))
    by_arm = {a["name"]: [r for r in rows if r["arm"] == a["name"]] for a in ARMS}

    # red-team F3: every arm must have encoded the SAME observation set per seed.
    shared_obs_ok = True
    for seed in seeds:
        hashes = {r["collection_obs_hash"] for r in rows if r["seed"] == seed}
        if len(hashes) != 1:
            shared_obs_ok = False
    assert shared_obs_ok, "collection observation set differs across arms for at least one seed"

    # CONSEQUENCE of the shared observation set plus the frozen-encoder finding: LINEAGE's
    # encoder never moves, so on an identical observation sequence LINEAGE and
    # UNTRAINED_ENCODER produce BIT-IDENTICAL latents. Measured per seed by comparing the
    # collected-state hashes. This is the design's own positive control -- it verifies the
    # shared-observation machinery AND the frozen-encoder claim end to end -- and it is why
    # T4 is a mechanical identity check rather than an independent baseline (it is kept in
    # the Bonferroni family anyway, which is the conservative choice).
    lin_hash = {r["seed"]: r["collection_latent_hash"] for r in by_arm[LOAD_BEARING_ARM]}
    unt_hash = {r["seed"]: r["collection_latent_hash"] for r in by_arm[BASELINE_ARM]}
    wrm_hash = {r["seed"]: r["collection_latent_hash"] for r in by_arm[ROUTING_ARM]}
    n_identical = sum(1 for s_ in lin_hash if unt_hash.get(s_) == lin_hash[s_])
    n_warmed_differs = sum(1 for s_ in lin_hash if wrm_hash.get(s_) != lin_hash[s_])

    # --- gates ---
    headroom_entries: Dict[str, Dict[str, Any]] = {}
    gates: Dict[str, Dict[str, Any]] = {}
    for unit in units:
        arm = unit["arm"]
        arm_rows = by_arm[arm]
        null_means = [r["probes"][LOAD_BEARING_FEATURES]["null_mean"] for r in arm_rows
                      if r["probes"][LOAD_BEARING_FEATURES]["null_mean"] == r["probes"][LOAD_BEARING_FEATURES]["null_mean"]]
        if null_means:
            headroom_entries[arm] = dv_headroom_check(
                f"dv_headroom_probe_excess_{arm}", dv_name="balanced accuracy excess over shuffle null (full64)",
                criterion_threshold=PROBE_MARGIN, control_values=null_means, statistic="ceiling_headroom",
                dv_bounds=PROBE_DV_BOUNDS, margin=1.0, arm=arm, null_means_per_seed=null_means)
        else:
            headroom_entries[arm] = {"name": f"dv_headroom_probe_excess_{arm}", "kind": "dv_headroom",
                                     "measured": float("nan"), "threshold": PROBE_MARGIN, "direction": "lower",
                                     "dv_name": "balanced accuracy excess (full64)",
                                     "achievable_statistic": "ceiling_headroom", "note": "no finite null means"}
        measured = {"heldout_adequate_seeds": float(sum(1 for r in arm_rows if r["heldout_adequate"])),
                    "dv_headroom_probe_excess": float(headroom_entries[arm]["measured"])}
        if unit["online_training"]:
            measured["writepath_engaged"] = float(min([r["n_write_calls_training"] for r in arm_rows], default=0))
        if unit["sd070_warmup"]:
            trained = [bool(r["sd070_warmup_report"].get("zworld_encoder_trained")) for r in arm_rows]
            measured["encoder_trained"] = float(sum(trained) / max(len(trained), 1))
        gates[arm] = evaluate_arm_gate(arm, unit, specs, measured=measured)
    gate = aggregate_arm_gates([gates[u["id"]] for u in units])

    # --- tests ---
    def one_sample(arm: str, fname: str) -> Dict[str, Any]:
        vals = [(r["probes"][fname]["excess"], r["seed"]) for r in by_arm[arm]
                if r["heldout_adequate"] and r["probes"][fname]["excess"] == r["probes"][fname]["excess"]]
        ex = [v for v, _ in vals]
        p = _permutation_test_pvalue(ex) if ex else float("nan")
        m = _mean(ex)
        green = gates[arm]["gate_green"]
        return {"arm": arm, "features": fname, "excess_per_seed": ex, "seeds": [s for _, s in vals],
                "n_seeds": len(ex), "mean_excess": m, "p_value": p, "required_margin": PROBE_MARGIN,
                "alpha_corrected": ALPHA_CORRECTED, "gate_green": green,
                "mean_balanced_accuracy": _mean([r["probes"][fname]["balanced_accuracy"] for r in by_arm[arm]]),
                "mean_null": _mean([r["probes"][fname]["null_mean"] for r in by_arm[arm]]),
                "passed": bool(green and m == m and m >= PROBE_MARGIN and p == p and p < ALPHA_CORRECTED),
                "non_degenerate": bool(green and len(ex) >= 2 and (max(ex) - min(ex)) > 1e-9)}

    tests: Dict[str, Any] = {
        "T1_lineage_zworld32": one_sample(LOAD_BEARING_ARM, LOAD_BEARING_FEATURES),
        "T2_sd070_warmed_zworld32": one_sample(ROUTING_ARM, LOAD_BEARING_FEATURES),
        "T4_untrained_encoder_zworld32": one_sample(BASELINE_ARM, LOAD_BEARING_FEATURES),
        # the stream as ContextMemory sees it -- reported with its own p, NOT in the
        # Bonferroni family and NOT a label input (red-team F1: z_self carries health /
        # harm_exposure, which falls to zero within 4 steps of every dangerous episode)
        "T1b_lineage_full64_stream_descriptive": one_sample(LOAD_BEARING_ARM, STREAM_FEATURES),
    }
    # T3: paired routing contrast on the z_world half
    lin = {r["seed"]: r["probe_excess_zworld32"] for r in by_arm[LOAD_BEARING_ARM]}
    wrm = {r["seed"]: r["probe_excess_zworld32"] for r in by_arm[ROUTING_ARM]}
    diffs = [(wrm[s] - lin[s], s) for s in lin if lin[s] is not None and wrm.get(s) is not None]
    d_vals = [d for d, _ in diffs]
    p3 = _permutation_test_pvalue(d_vals) if d_vals else float("nan")
    m3 = _mean(d_vals)
    # red-team F2: T3's DV is bounded by 1 - acc(LINEAGE); certify headroom above the
    # lineage's REALISED accuracy, not above the null.
    lin_acc = [r["probes"][LOAD_BEARING_FEATURES]["balanced_accuracy"] for r in by_arm[LOAD_BEARING_ARM]
               if r["heldout_adequate"] and r["probes"][LOAD_BEARING_FEATURES]["balanced_accuracy"] == r["probes"][LOAD_BEARING_FEATURES]["balanced_accuracy"]]
    if lin_acc:
        headroom_entries["T3_above_lineage_accuracy"] = dv_headroom_check(
            "dv_headroom_T3_above_lineage_accuracy", dv_name="paired excess diff SD070_WARMED - LINEAGE (zworld32)",
            criterion_threshold=PROBE_MARGIN, control_values=lin_acc, statistic="ceiling_headroom",
            dv_bounds=PROBE_DV_BOUNDS, margin=1.0, control_arm=LOAD_BEARING_ARM, lineage_accuracy_per_seed=lin_acc)
        t3_headroom_met = float(headroom_entries["T3_above_lineage_accuracy"]["measured"]) > PROBE_MARGIN
    else:
        headroom_entries["T3_above_lineage_accuracy"] = {"name": "dv_headroom_T3_above_lineage_accuracy",
                                                         "kind": "dv_headroom", "measured": float("nan"),
                                                         "threshold": PROBE_MARGIN, "direction": "lower",
                                                         "dv_name": "paired excess diff (zworld32)",
                                                         "achievable_statistic": "ceiling_headroom"}
        t3_headroom_met = False
    # red-team F7: the paired count must keep the sign-flip floor under alpha.
    t3_pairs_ok = len(d_vals) >= MIN_ADEQUATE_SEEDS
    green3 = bool(gates[LOAD_BEARING_ARM]["gate_green"] and gates[ROUTING_ARM]["gate_green"]
                  and t3_headroom_met and t3_pairs_ok)
    tests["T3_recipe_raises_decodability"] = {
        "paired_diffs": d_vals, "seeds": [s for _, s in diffs], "n_paired_seeds": len(d_vals),
        "p_floor": (0.5 ** len(d_vals)) if d_vals else float("nan"), "pairs_floor_ok": t3_pairs_ok,
        "headroom_above_lineage_accuracy_met": t3_headroom_met,
        "mean_diff": m3, "p_value": p3, "required_margin": PROBE_MARGIN, "alpha_corrected": ALPHA_CORRECTED,
        "gate_green": green3,
        "passed": bool(green3 and m3 == m3 and m3 >= PROBE_MARGIN and p3 == p3 and p3 < ALPHA_CORRECTED),
        "non_degenerate": bool(green3 and len(d_vals) >= 2 and (max(d_vals) - min(d_vals)) > 1e-9)}
    descriptive = {arm: {f: {"mean_excess": _mean([r["probes"][f]["excess"] for r in by_arm[arm] if r["heldout_adequate"]]),
                             "mean_balanced_accuracy": _mean([r["probes"][f]["balanced_accuracy"] for r in by_arm[arm]])}
                         for f in FEATURE_SETS} for arm in by_arm}

    t1, t2, t3 = tests["T1_lineage_zworld32"], tests["T2_sd070_warmed_zworld32"], tests["T3_recipe_raises_decodability"]
    t4 = tests["T4_untrained_encoder_zworld32"]
    g_lin, g_wrm = gates[LOAD_BEARING_ARM]["gate_green"], gates[ROUTING_ARM]["gate_green"]
    if not (g_lin or g_wrm):
        label = "probe_gate_not_ready"
    elif not g_lin:
        label = f"lineage_unscored_sd070_warmed_{'decodable' if t2['passed'] else 'at_chance'}"
    elif t1["passed"]:
        label = "lineage_stream_linearly_decodable_structure_off_raw_axis"
    elif t2["passed"] and t3["passed"]:
        label = "lineage_stream_at_chance_sd070_recipe_restores_decodability"
    elif t2["passed"]:
        label = "lineage_stream_at_chance_warmed_decodable_contrast_underpowered"
    elif g_wrm:
        label = "write_stream_at_chance_under_both_regimes"
    else:
        label = "lineage_stream_at_chance_sd070_warmed_unscored"
    overall_pass = bool(t1["passed"])
    status = "PASS" if overall_pass else "FAIL"
    evidence_direction = "supports" if t3["passed"] else "non_contributory"
    non_degenerate = g_lin

    if g_lin:
        adjudication_preconditions = gate["adjudication_preconditions"]
        degeneracy_reason = gate["degeneracy_reason"]
    else:
        adjudication_preconditions = list(gates[LOAD_BEARING_ARM]["preconditions"])
        degeneracy_reason = (f"LINEAGE (load-bearing arm) gate RED: {', '.join(gates[LOAD_BEARING_ARM]['failed_preconditions'])} "
                             f"-- the load-bearing arm was not scored; " + gate["degeneracy_reason"])

    lineage_frozen_cells = [r["online_latent_stack_delta"].get("n_changed", 0) == 0 for r in by_arm[LOAD_BEARING_ARM]]
    warmed_online_frozen = [r["online_latent_stack_delta"].get("n_changed", 0) == 0 for r in by_arm[ROUTING_ARM]]
    lineage_encoder_finding = {
        "name": "lineage_encoder_frozen_by_construction", "load_bearing": False, "kind": "finding",
        "passed": all(lineage_frozen_cells) if lineage_frozen_cells else False,
        "lineage_cells_with_zero_latent_stack_delta": sum(lineage_frozen_cells), "lineage_cells": len(lineage_frozen_cells),
        "sd070_warmed_online_phase_cells_with_zero_delta": sum(warmed_online_frozen), "sd070_warmed_cells": len(warmed_online_frozen),
        "note": ("Measured at authoring time (2026-09-07) and re-measured here: the 956/970/972 harness trains "
                 "E1/E2/E3/write_addr_tagger on DETACHED buffered latents, so no gradient reaches latent_stack "
                 "and the write stream every ContextMemory content-half experiment measured is a random "
                 "projection of the observation. If every LINEAGE cell reads zero delta, V3-EXQ-972's 0.028 "
                 "cosine and V3-EXQ-970a's tagger input are both frozen-encoder streams -- read T4 (untrained "
                 "encoder) as the same encoder on a different trajectory."),
    }
    identity_control = {
        "name": "lineage_equals_untrained_encoder_identity_control", "load_bearing": False, "kind": "control",
        "passed": bool(n_identical == len(lin_hash) and n_warmed_differs == len(lin_hash)),
        "n_seeds_lineage_identical_to_untrained": n_identical, "n_seeds": len(lin_hash),
        "n_seeds_sd070_warmed_differs": n_warmed_differs,
        "note": ("Positive control on BOTH the shared-observation-set fix and the frozen-encoder "
                 "finding: with every arm encoding the same observation sequence, LINEAGE (whose "
                 "encoder no gradient reaches) must produce latents bit-identical to "
                 "UNTRAINED_ENCODER, while SD070_WARMED (whose P0a moved the world encoder) must "
                 "differ. A failure of either half means the shared-observation machinery or the "
                 "frozen-encoder claim is wrong, and T3 is not an encoder contrast."),
    }
    criteria = [
        {"name": "T1_lineage_write_stream_linearly_decodable", "load_bearing": True, "passed": t1["passed"],
         "mean_excess": t1["mean_excess"], "p_value": t1["p_value"], "n_seeds": t1["n_seeds"],
         "combination_rule": (f"LINEAGE gate green AND mean per-seed (balanced accuracy - shuffle-null mean) "
                              f">= {PROBE_MARGIN} AND exact sign-flip p < {ALPHA_CORRECTED} (Bonferroni over 4 tests)"),
         "note": "972's question held out: a FAIL means the lineage's write stream carries no linearly "
                 "decodable context at this n -- read with T2/T3 for the SD-070 routing, and T4 for what the "
                 "observation hands a random projection."},
        {"name": "T2_sd070_warmed_write_stream_linearly_decodable", "load_bearing": False, "kind": "secondary",
         "passed": t2["passed"], "mean_excess": t2["mean_excess"], "p_value": t2["p_value"], "n_seeds": t2["n_seeds"]},
        {"name": "T3_sd070_recipe_raises_decodability", "load_bearing": False, "kind": "routing",
         "passed": t3["passed"], "mean_diff": t3["mean_diff"], "p_value": t3["p_value"], "n_paired_seeds": t3["n_paired_seeds"],
         "note": "paired SD070_WARMED - LINEAGE; the SD-070 evidence_direction reads 'supports' iff this passes"},
        {"name": "T4_untrained_encoder_write_stream_linearly_decodable", "load_bearing": False, "kind": "baseline",
         "passed": t4["passed"], "mean_excess": t4["mean_excess"], "p_value": t4["p_value"]},
        {"name": "T1b_lineage_full64_write_stream_descriptive", "load_bearing": False, "kind": "descriptive",
         "passed": tests["T1b_lineage_full64_stream_descriptive"]["passed"],
         "mean_excess": tests["T1b_lineage_full64_stream_descriptive"]["mean_excess"],
         "p_value": tests["T1b_lineage_full64_stream_descriptive"]["p_value"],
         "note": "the stream as ContextMemory.write() receives it; NOT a label input and NOT in the "
                 "Bonferroni family: the z_self half encodes health / harm_exposure, which separate "
                 "the classes by interoception (red-team F1), so a high value here is expected and "
                 "says nothing about the z_world encoder. Compare with zself32 in descriptive.per_arm."},
        lineage_encoder_finding,
        identity_control,
        {"name": "descriptive_half_probes_and_972_readacross", "load_bearing": False, "kind": "descriptive",
         "per_arm": descriptive,
         "cosine_972_style_mean_per_arm": {a: _mean([r["cosine_separability_972_style"]["separability_score"] for r in by_arm[a]]) for a in by_arm},
         "zworld_participation_ratio_mean_per_arm": {a: _mean([r["participation_ratio"]["zworld_participation_ratio"] for r in by_arm[a]]) for a in by_arm}},
    ]
    criteria_non_degenerate = {
        "T1_lineage_write_stream_linearly_decodable": non_degenerate and t1["non_degenerate"],
        "T2_sd070_warmed_write_stream_linearly_decodable": t2["non_degenerate"],
        "T3_sd070_recipe_raises_decodability": t3["non_degenerate"],
        "T4_untrained_encoder_write_stream_linearly_decodable": t4["non_degenerate"],
    }

    metrics = {
        "per_seed_probe_excess_zworld32": {a: [r["probe_excess_zworld32"] for r in by_arm[a]] for a in by_arm},
        "per_seed_probe_excess_full64": {a: [r["probe_excess_full64"] for r in by_arm[a]] for a in by_arm},
        "shared_observation_set_verified": shared_obs_ok,
        "lineage_untrained_latent_identity": {"n_identical": n_identical, "n_seeds": len(lin_hash),
                                              "n_warmed_differs": n_warmed_differs},
        "warmup_env": "alternating safe/dangerous (departs from the 1006 lineage's safe-only integration; red-team F5)",
        "per_seed_probe_balanced_accuracy": {a: {f: [r["probes"][f]["balanced_accuracy"] for r in by_arm[a]] for f in FEATURE_SETS} for a in by_arm},
        "per_seed_probe_null_mean": {a: {f: [r["probes"][f]["null_mean"] for r in by_arm[a]] for f in FEATURE_SETS} for a in by_arm},
        "per_seed_n_test_per_class": {a: [r["probe_split"]["n_test_per_class"] for r in by_arm[a]] for a in by_arm},
        "per_seed_n_write_calls_training": {a: [r["n_write_calls_training"] for r in by_arm[a]] for a in by_arm},
        "per_seed_zworld_participation_ratio": {a: [r["participation_ratio"]["zworld_participation_ratio"] for r in by_arm[a]] for a in by_arm},
        "per_seed_cosine_separability_972_style": {a: [r["cosine_separability_972_style"]["separability_score"] for r in by_arm[a]] for a in by_arm},
        "per_seed_class_latent_norms": {a: [r["class_latent_norms"] for r in by_arm[a]] for a in by_arm},
        "tests": tests, "descriptive": descriptive, "dv_headroom_entries": headroom_entries, "gate_audit": gate_audit,
        # Flat scalar readout, MERGED INTO this `metrics` dict rather than emitted as a
        # sibling `readout` (REE_assembly evidence/planning/
        # flat_scalar_readout_recording_gap_20260909.md). The runpack converter takes the
        # FIRST non-empty of metrics / aggregates / summary_metrics / readout, so this
        # already-populated `metrics` would shadow a `readout` sibling entirely -- and
        # every entry above is keyed by arm, feature set or test, so it harvested zero
        # NUMERIC entries and the pack scored empty: no fail_if stop threshold could fire,
        # the duplicate-emission supersession fingerprint was skipped, and the index
        # carried no deltas. The nested per-arm/per-test blocks are kept unchanged beside
        # these scalars.
        #
        # PASS is carried by T1 alone, but the SD-070 evidence direction is routed by T3,
        # so both are recorded independently -- a run can be FAIL on T1 and still
        # `supports` SD-070 via T3, and a single pass bit would lose that. Each test's
        # gate_green is recorded beside its pass flag for the same reason the labels
        # distinguish them: an ungated test is UNSCORED, not refuted. T3's two red-team
        # preconditions (headroom above the LINEAGE's realised accuracy, not above the
        # null; and the paired count keeping the sign-flip floor under alpha) are recorded
        # too, since T3 green depends on both. flat_readout() enforces the two encoding
        # rules (bools -> 0/1 ints; non-finite/None dropped -- and mean_diff / p_value are
        # genuinely NaN when a test had no adequate pairs, so dropping them is correct).
        # Recording-only: the verdict grid, criteria, thresholds and DVs are unchanged.
        **flat_readout(dict(
            {
                "T1_lineage_zworld32_passed": t1["passed"],
                "T2_sd070_warmed_zworld32_passed": t2["passed"],
                "T3_recipe_raises_decodability_passed": t3["passed"],
                "T4_untrained_encoder_zworld32_passed": t4["passed"],
                "overall_pass_flag": overall_pass,
                "sd070_supports_flag": t3["passed"],
                "non_degenerate_flag": non_degenerate,
                "lineage_arm_gate_green": g_lin,
                "routing_arm_gate_green": g_wrm,
                "probe_margin": PROBE_MARGIN,
                "alpha_corrected": ALPHA_CORRECTED,
                "min_adequate_seeds": MIN_ADEQUATE_SEEDS,
                # T3's two red-team preconditions
                "t3_headroom_above_lineage_accuracy_met": t3_headroom_met,
                "t3_pairs_floor_ok": t3_pairs_ok,
                # lineage identity control
                "lineage_untrained_latent_n_identical": n_identical,
                "lineage_untrained_latent_n_seeds": len(lin_hash),
                "lineage_untrained_latent_n_warmed_differs": n_warmed_differs,
                "shared_observation_set_verified": shared_obs_ok,
                "n_cells": len(rows),
            },
            **{
                f"{_k}_{_f}": _t[_f]
                for _k, _t in tests.items()
                for _f in ("mean_excess", "mean_diff", "p_value", "n_paired_seeds",
                           "gate_green", "passed", "non_degenerate")
                if _f in _t
            },
        )),
    }

    lines = [f"# {QUEUE_ID} -- SD-070 held-out linear probe of the ContextMemory write stream (972 successor)", "",
             f"**Status:** {status}  **Label:** {label}  **SD-070 direction:** {evidence_direction}", "",
             "| test | mean excess / diff | p | gate | pass |", "|---|---|---|---|---|"]
    for k, t in tests.items():
        v = t.get("mean_excess", t.get("mean_diff"))
        lines.append(f"| {k} | {v:.3f} | {t['p_value']:.4f} | {t['gate_green']} | {t['passed']} |")
    lines += ["", f"margin {PROBE_MARGIN}, alpha_corrected {ALPHA_CORRECTED}", degeneracy_reason or ""]

    result: Dict[str, Any] = {
        "outcome": status, "status": status, "claim_ids": CLAIM_IDS, "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE, "evidence_direction": evidence_direction,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "hypothesis_space": {"qid": "contextmemory_write_content_discrimination", "hid": "H4-input-distribution",
                             "note": "972 successor; routed to SD-070 per the substrate entry's hint (2)"},
        "metrics": metrics, "arm_results": rows, "summary_markdown": "\n".join(lines),
        "per_arm_gate": gate["per_arm_gate"], "non_degenerate": non_degenerate, "degeneracy_reason": degeneracy_reason,
        "interpretation": {"label": label, "preconditions": adjudication_preconditions,
                           "criteria_non_degenerate": criteria_non_degenerate, "criteria": criteria,
                           "sd070_routing": {"T3_passed": t3["passed"], "evidence_direction": evidence_direction}},
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
        for row in result["arm_results"]:
            assert row["n_collect_records"] > 0, f"{row['arm']}: no collected states"
            assert row["probes"]["zworld32"]["n_test"] > 0, f"{row['arm']}: probe never evaluated"
            assert row["class_latent_norms"]["dangerous_z_self"] == row["class_latent_norms"]["dangerous_z_self"]
            if row["sd070_warmup"]:
                assert row["sd070_warmup_report"].get("p0a_ran") is True, "SD-070 P0a did not run at smoke scale"
                assert row["sd070_warmup_report"].get("guard_checked") is True
            if row["online_training"]:
                assert row["online_latent_stack_delta"]["checked"]
        assert result["metrics"]["shared_observation_set_verified"], (
            "collection observation set not shared across arms")
        _id = result["metrics"]["lineage_untrained_latent_identity"]
        assert _id["n_identical"] == _id["n_seeds"], (
            "LINEAGE latents differ from UNTRAINED_ENCODER despite a shared observation set and a "
            "frozen encoder -- the shared-observation machinery or the frozen-encoder claim is wrong")
        assert _id["n_warmed_differs"] == _id["n_seeds"], (
            "SD070_WARMED latents are identical to LINEAGE -- the P0a warmup did not change the "
            "encoder's output, so T3 measures nothing")
        print("[smoke] every arm collected step-matched states, evaluated the held-out probe, and the "
              "SD-070 P0a trainer ran under the encoder guard", flush=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["timestamp_utc"] = ts
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    full_config = {
        "arms": ARMS, "seeds": SEEDS, "training_episodes": TRAINING_EPISODES, "collect_episodes": COLLECT_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE, "context_switch_every": CONTEXT_SWITCH_EVERY,
        "zworld_p0_episodes": ZWORLD_P0_EPISODES, "step_match_max": STEP_MATCH_MAX,
        "write_selection": WRITE_SELECTION, "write_addressing_loss_weight": WRITE_ADDRESSING_LOSS_WEIGHT,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE, "alpha_world": ALPHA_WORLD, "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS, "latent_dim": LATENT_DIM, "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "probe_train_frac": PROBE_TRAIN_FRAC, "min_test_per_class": MIN_TEST_PER_CLASS,
        "load_bearing_features": LOAD_BEARING_FEATURES, "collect_env_seed_offset": COLLECT_ENV_SEED_OFFSET,
        "collect_policy_seed_offset": COLLECT_POLICY_SEED_OFFSET, "warmup_env": "alternating_safe_dangerous",
        "min_adequate_seeds": MIN_ADEQUATE_SEEDS, "n_probe_shuffle": N_PROBE_SHUFFLE, "probe_l2": PROBE_L2,
        "probe_lbfgs_iters": PROBE_LBFGS_ITERS, "probe_margin": PROBE_MARGIN, "alpha_corrected": ALPHA_CORRECTED,
        "n_tests": N_TESTS, "write_calls_floor": WRITE_CALLS_FLOOR, "sep_subsample_max": SEP_SUBSAMPLE_MAX,
    }
    out_path = write_flat_manifest(result, dry_run=args.dry_run, config=full_config, seeds=SEEDS,
                                   script_path=__file__, started_at=t0,
                                   z_goal_stream_stats=zg_accumulator.stats())
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    emit_outcome(outcome=result["status"] if result["status"] in ("PASS", "FAIL") else "FAIL",
                 manifest_path=out_path, dry_run=args.dry_run)
