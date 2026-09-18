#!/opt/local/bin/python3
"""
V3-EXQ-1057 -- MECH-017 reality consolidation: is the RECENCY COST of offline replay
INTRINSIC to replay, or an artefact of REALLOCATING a fixed gradient budget? The
three V3-EXQ-1048 arms, unchanged, plus a fourth at ADDITIVE budget.

SLEEP DRIVER: N/A -- no SleepLoopManager is built (use_sleep_loop / sws_enabled /
              rem_enabled / use_sleep_aggregation_cluster are all left at their
              default False). The manipulation is the MECH-423 R3
              CrossModuleConsolidator pass called DIRECTLY (the same call shape
              V3-EXQ-680e validated), deliberately outside a sleep cycle -- see
              WHY NO SLEEP CYCLE below.

THE QUESTION, AND WHERE IT CAME FROM
-------------------------------------
V3-EXQ-1048 ran this instrument on 2026-09-17 and FAILED its C2 leg: at MATCHED
gradient budget, offline replay (A) beat the budget-matched online comparator (B)
on the EARLY probes by a large margin (C1 5/5, mean relative gain 0.4542 against a
0.05 floor) but was WORSE than B on the LATE probes on 5/5 seeds, by +9.25% to
+95.13% (mean +41.4%) against LATE_TOL = 0.10. The effect replicated on the
untouched E2 self-forward channel (A 11.9% worse). All 12 preconditions were green
and the substrate performed as designed.

Its CONFIRMED failure autopsy
(REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1048_2026-09-17.json,
routing `queue-experiment`, recommended_substrate_queue_entry.action `none`) names
this successor verbatim in learning_extracted[0]:

    "The separable successor question is whether the recency deficit is INTRINSIC
     to replay or an artefact of REALLOCATING a fixed budget; that needs an arm
     with ADDITIONAL rather than reallocated budget, and it is a NEW question, not
     a re-spec of C2."

That is this run. Under a matched budget every step spent replaying old traces is
a step NOT spent on recent ones, so a recency cost is exactly what a pure
reallocation would produce even if replay itself is harmless to recency. The two
hypotheses are only separable by ADDING the replay pass to the online budget
instead of substituting it.

  ARM_A_OFFLINE_REPLAY        -- N consolidation gradient steps per point, drawn
                                 over the WHOLE life buffer (old + recent states).
  ARM_B_BUDGET_MATCHED_RECENT -- the IDENTICAL call, identical N, but the buffers
                                 visible to the pass are narrowed to the RECENT
                                 window (the online stream since the last point).
                                 Same compute, no replay of old states.
  ARM_C_NO_EXTRA_TRAINING     -- no extra steps at all (the fidelity reference).
  ARM_D_ADDITIVE_BUDGET       -- THE NEW ARM. At each consolidation point it runs
                                 ARM_A's whole-buffer pass AND ARM_B's
                                 recent-window pass, as two INDEPENDENT sequential
                                 `consolidator.consolidate(...)` calls: 2 x
                                 CMC_STEPS total, i.e. A's budget PLUS B's. Replay
                                 is ADDED to the online budget, not substituted
                                 for it.

CONFIRMING (the recency cost was a REALLOCATION ARTEFACT):
    e1_holdout_mse_late(D) <= (1 + LATE_TOL) * e1_holdout_mse_late(B) on at least
    SIGN_CONSISTENCY_REQUIRED of the seeds -- the deficit DISAPPEARS once the
    budget is added rather than reallocated.
FALSIFYING (the recency cost is INTRINSIC to replay):
    it does not -- D is still worse than B by more than LATE_TOL at the same
    majority, so replaying stale traces costs recency even when it takes nothing
    away from the online stream.

CRITERION PROVENANCE -- NOTHING NEW WAS INVENTED. C4 is V3-EXQ-1048's C2 with ARM_D
substituted for ARM_A: same DV (`e1_holdout_mse_late`), same comparator arm (B),
same LATE_TOL = 0.10, same SIGN_CONSISTENCY_REQUIRED = 4. The additive budget is
2 x CMC_STEPS because that is the only value derivable as "A's budget plus B's".
Re-pointing the late leg at ARM_C is explicitly FORBIDDEN: the autopsy's
learning_extracted[1] records that its own first draft did exactly that and was
wrong -- "the claim itself names arm B on that leg, and arm C only as a readiness
control", and the swap "would not even have rescued C2 (3/5 against a threshold
of 4)".

SCOPE -- WHAT THIS RUN DOES NOT ADJUDICATE. The same autopsy records that MECH-017
"is two claims wearing one id": 'replay counters forgetting' is SUPPORTED, 'at no
cost to recency' is WEAKENED, and the id was flagged for a future granularity
review. This run answers the REALLOCATION question about the recency conjunct and
nothing else. It must not be read as adjudicating the bundle. The matched-budget
legs C1/C2/C3 are re-run unchanged but carry `load_bearing: false` -- they are
replication of 1048 on the same harness, and cannot move this run's verdict.

WHAT IS MANIPULATED, EXACTLY (attribution)
-------------------------------------------
A and B run the SAME live substrate function
(`CrossModuleConsolidator.consolidate(module_losses={e1: agent.compute_prediction_loss,
e2: agent.compute_e2_loss}, ...)`, ree_core/sleep/cross_module_consolidation.py),
with the SAME `n_steps`, the SAME schedule, the SAME lr, and a fresh per-module
Adam built inside each call (so no optimizer-state asymmetry). The ONLY
difference is which slice of `agent._{self,world,action}_experience_buffer` /
`agent._e2_transition_buffer` is visible to the loss closures for the duration of
the call: the whole buffer (A) or its last RECENT_WINDOW entries (B). That is
"replay vs no replay, at matched gradient budget" with nothing else moving.

Waking training is IDENTICAL in all four arms: after every training episode each
arm runs WAKE_STEPS of the same pass restricted to THAT episode's entries (the
online stream). This is what produces the recency pressure the claim's premise
needs; it is not part of the contrast.

WHY NO SLEEP CYCLE (and why that is the stronger design)
---------------------------------------------------------
Routing the manipulation through `SleepLoopManager.force_cycle()` would make the
buffer narrowing in arm B also visible to every OTHER buffer reader inside
`_run_cycle` -- `offline_integration()`'s `e1.integrate_experience` (a second E1
weight update), the SWS schema pass, REM attribution, the replay sampler -- so an
A-vs-B difference would no longer be attributable to the consolidation pass. The
sleep-cycle WIRING of this exact consolidator is separately validated by
V3-EXQ-1026 (PASS, 2026-09-14: the hook fires and moves E2 through a real cycle),
so re-exercising it here buys nothing and costs attribution. This run isolates the
consolidation CONTENT; 1026 owns the wiring.

TWO SUBSTRATE LIMITS, DECLARED UP FRONT (not worked around)
-------------------------------------------------------------
1. MECH-017's `what_would_answer` names "held-out one-step E2 world_forward
   error" as a co-primary readout. That readout is STRUCTURALLY UNAVAILABLE on
   the current substrate: open substrate_queue entry `e2-world-forward-sleep-trainer`
   (status pending_implementation, severity degrading, substrate_paths
   ree_core/sleep/cross_module_consolidation.py + phase_manager.py) records that
   `consolidate()` updates only E2's SELF-forward head and that `E2.world_forward`
   has no trainer anywhere in ree_core -- so a world_forward DV is identically 0.0
   in EVERY arm by construction. This run therefore does NOT attempt it. The E2
   readout reported below is E2 SELF-forward (`predict_next_self`), recorded as a
   SECONDARY, explicitly NON-load-bearing diagnostic and labelled as such, so it
   cannot be mistaken for the world_forward quantity the claim asks for.
   The load-bearing verdict rests entirely on the E1 leg, which MECH-017 names as
   its other primary readout ("multi-step E1 error") and which IS trained by this
   pass (`updates_e1`, asserted as a precondition).
2. The open CORRUPTING entry `contextmemory-write-path-addressing-degeneracy`
   (substrate_paths `ree_core/predictors/e1_deep.py::ContextMemory.write`) is
   OFF-PATH here, not argued away: `ContextMemory.write()` is reached only from
   `E1Deep.update_from_observation`, gated on `E1Config.sd016_writepath_mode`,
   which this run leaves at its default `"off"` (config.py:631). The defective
   function is therefore never called. A precondition asserts the mode is "off"
   at runtime rather than trusting the default.

WHY THE ENCODER IS FROZEN (deliberate, and load-bearing for the design)
------------------------------------------------------------------------
Nothing in this driver trains `agent.latent_stack`: `CrossModuleConsolidator`
builds optimizers over `agent.e1.parameters()` / `agent.e2.parameters()` only.
That is the design, not an SD-070 omission (`_train_all_on_agent` /
`zworld_p0_episodes` are not used here at all): a trained encoder would make
z_self/z_world ARM-DEPENDENT, and the claim requires a FIXED probe set shared by
all arms. With the encoder frozen and the action stream model-independent (see
below), every arm sees bit-identical latents -- asserted, not assumed, by the
`probe_set_identical_across_arms` precondition, which hashes each cell's probe
tensors and requires all four arms of a seed to agree exactly.
`alpha_world` / `alpha_self` are raised to 0.9 (from the 0.3 default) so z_world
tracks the observation instead of being heavily EMA-smoothed -- at 0.3 the
forward-prediction task degenerates toward persistence.

This IS the mandatory phased-training discipline, in its strictest form, not an
exemption from it: P0 = the WARMUP_EPISODES encoder-only rollout (no downstream
loss); P1 = every E1/E2 gradient step in this run, taken over
`_world_experience_buffer` / `_e2_transition_buffer` entries that were stored as
`latent_state.z_*.detach().clone()` (agent.py) against a permanently frozen
encoder -- so the heads never chase a moving latent target (the EXQ-166b/c/d,
EXQ-085l, EXQ-194 failure mode); P2 = the held-out probe evaluation, under
`torch.no_grad()`.

MODEL-INDEPENDENT BEHAVIOUR (commitment-free by construction)
---------------------------------------------------------------
Actions are drawn from a per-cell `torch.Generator(seed)`, never from
`agent.select_action`. This is deliberate: it makes the experience stream
identical across arms (so the arms cannot diverge behaviourally, and the probe
set really is fixed), and it keeps the run entirely clear of the E3/F-dominance
committed-selection layer -- there is no action-commitment DV here, so the known
conversion ceiling cannot manufacture or mask this result.

The claim's precondition (iv) -- "waking behaviour must vary across the life" --
is supplied structurally instead: the EARLY regime starts every episode at
EARLY_START with an up/left action bias, the LATE regime at LATE_START with a
down/right bias, and the realized grid-cell occupancy divergence between the two
regimes is MEASURED as a precondition, not assumed.

NON-DEGENERACY PRECONDITIONS (all four of the claim's, operationalized + measured)
-----------------------------------------------------------------------------------
 (i)   `forgetting_present_in_control` -- in ARM_C, the SAME early probe set
       evaluated at the end of the life must have degraded by >= FORGETTING_FLOOR
       against its own value measured immediately after capture (before the late
       block). Deliberately WITHIN-STRATUM: an early-vs-late comparison would be
       confounded by intrinsic regime difficulty, since the two regimes occupy
       different parts of the grid (at full scale ARM_A's early probes are the
       EASIER ones). If the waking model has not forgotten the early states there
       is nothing for replay to counter and all arms tie (the claim's "world model
       must NOT be saturated"). Read off arm C only; the verdict is A-vs-B, so
       this gate does not certify its own subject.
 (ii)  `replay_covers_early_regime` -- in ARM_A, the share of the buffer visible
       to the consolidation pass that was recorded during the EARLY regime,
       averaged over the LATE-block consolidation points. `compute_prediction_loss`
       draws `start_idx ~ Uniform[0, buf_len-1)`, so this buffer share IS the
       per-draw probability of an early-regime trace. Arm B's value is recorded
       alongside (0.0 by construction) -- that contrast IS the manipulation.
 (iii) `consolidation_updated_e1` / `consolidation_updated_e2` -- min over ARM_A
       cells of the pass's own `updates_e1` / `updates_e2` counters must be >= 1,
       i.e. the pass really moved the modules the DV reads.
 (iv)  `early_late_state_divergence` -- Jaccard DISTANCE between the sets of grid
       cells occupied in the early regime and in the late regime, floored. If old
       and recent states coincide there is nothing for replay to add
       (sleep_substrate:GAP-2's bit-identical-arms vacuous case).
Plus five instrument gates: `replay_window_separation` (A's and B's replay windows
must actually differ -- the MANIPULATION itself), `manipulation_reaches_dv` (A and B
must not produce a bit-identical DV; a dead manipulation is not a null result),
`e1_has_skill_over_persistence` (E1 must beat a do-nothing forward model, else a
"fidelity" comparison over it means nothing), `probe_set_identical_across_arms` (the
probe set is genuinely FIXED), `probe_target_variance` (the probe targets actually
move), and `sd016_writepath_mode_off` (limit 2 above).

The first two were added after this script's FIRST smoke run: --dry-run shrank the
episode length while RECENT_WINDOW stayed at its full-run constant, the window
exceeded the whole buffer, and arms A and B came out bit-identical -- i.e. a dead
manipulation that would have read as the claim's "A == B" falsification. The window
is now derived from the episode length in use (`_recent_window`), and both states
are gated.

Any unmet precondition routes the whole run to `substrate_not_ready_requeue` with
`non_degenerate: false` -- never to a MECH-017 verdict.

DV-SYMMETRY INVARIANCE (per arm, per the mandatory declaration)
----------------------------------------------------------------
Every arm's DV is a held-out MSE -- `F.mse_loss(E1.predict_long_horizon(...),
targets)` on a FIXED, frozen target tensor -- i.e. a real-valued function of the
E1/E2 WEIGHTS evaluated at fixed inputs.
  - ARM_A / ARM_B: the manipulation is WHICH TRACES the gradient steps are
    computed on. It changes the gradient direction and hence the weights; MSE at
    fixed inputs is not invariant under a change of weights. It is not a broadcast
    additive constant (nothing is added to the DV), not a monotone rescaling
    (nothing rescales the DV), and not a permutation of interchangeable units (the
    probe windows are a fixed ordered set; the manipulation acts on the TRAINING
    draw, not on the probe set). None of the three invariance classes applies.
  - ARM_C: the manipulation is the ABSENCE of those steps -- a code-path gate, not
    a value transform of the DV. Same reading.
  - ARM_D: the manipulation is the ADDITION of a second, differently-windowed pass.
    It is a change in the NUMBER and COMPOSITION of gradient steps, so it moves the
    weights; same reading as A/B. Note it is emphatically NOT an additive constant
    on the DV -- nothing is added to the measured MSE, only to the training budget,
    and the two are related through the optimizer, not by arithmetic.
  The DV is also not a rank/argmax statistic, so the monotone-transform class is
  inapplicable to all four arms.

RED-TEAM (Step 4.5)
--------------------
INHERITED, from V3-EXQ-1048 (model: fable). Three findings, all CONFIRMED and all
FIXED in the code this script is derived from; the fixes are carried here unchanged
and are load-bearing for this run too:
  F1 (Family 3) -- the verdict grid's `else` catch-all absorbed a criteria TRADE-OFF
     and a budget-mismatch INSTRUMENT failure into one label. Fixed by moving the
     gradient-budget match OUT of the criteria into the preconditions. THIS run
     extends that same fix to the new arm: `additive_budget_is_a_plus_b` is a
     PRECONDITION, so a mis-dosed ARM_D routes to substrate_not_ready_requeue rather
     than being scored as a MECH-017 verdict.
  F2 (Family 4) -- the persistence-skill gate read ARM_C only, the one arm the
     verdict never uses. Re-scoped to the compared arms as a per-seed max. Carried
     unchanged: it still ranges over A and B, and is NOT widened to include ARM_D,
     because a max over three arms would be strictly WEAKER than the gate 1048
     passed.
  F3 (provenance) -- `reset_all_rng(seed)` runs in `arm_cell.__enter__`, which
     `main()` applies and a direct `run_cell(...)` call does NOT. Any figure taken
     off-wrapper is off-contract. No off-wrapper figure is quoted in this docstring.

THIS RUN'S OWN RED-TEAM: see the queue entry note.

THE PRIOR, FROM RECORDED EVIDENCE
----------------------------------
Unlike V3-EXQ-1048, this design needs no scratch pre-runs: its A/B/C arms are the
SAME arms, and their landing zone is already a RECORDED, citable manifest --
evidence/experiments/v3_exq_1048_mech017_reality_consolidation_replay_vs_budget_matched/
runs/..._20260917T102244Z_v3/ (outcome FAIL, evidence_direction `mixed`, autopsied
and CONFIRMED). From it: C1 5/5, C3 mean relative gain 0.4542 (floor 0.05), C2 0/5
with A worse than B on the late probes by +9.25% to +95.13% (mean +41.4%), all 12
preconditions green, 96 consolidator updates per cell. So C1/C2/C3 here are a
REPLICATION with a known expected outcome, which is exactly why they are not
load-bearing.

ARM_D IS GENUINELY UNRUN AND ITS DIRECTION IS UNKNOWN. Nothing in the recorded
manifest constrains it: no prior manifest anywhere carries an additive-budget arm,
so `e1_holdout_mse_late(D)` has never been measured on any substrate. Both branches
of C4 are live, and neither is the "expected" one:
  - D within tolerance of B  -> the deficit was the opportunity cost of reallocated
    compute. MECH-017's recency conjunct survives, restricted to the additive case.
  - D still worse than B     -> the deficit is a property of replaying stale traces
    as such. A genuine weakening, and a much stronger result than 1048's, because
    the obvious confound has been removed.
No threshold in this script was chosen or moved with any knowledge of D's value.

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1057_mech017_reality_consolidation_replay_additive_budget.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1057_mech017_reality_consolidation_replay_additive_budget.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.sleep.cross_module_consolidation import (
    CrossModuleConsolidator,
    CrossModuleConsolidatorConfig,
)
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest


EXPERIMENT_TYPE = "v3_exq_1057_mech017_reality_consolidation_replay_additive_budget"
QUEUE_ID = "V3-EXQ-1057"
CLAIM_IDS: List[str] = ["MECH-017"]
EXPERIMENT_PURPOSE = "evidence"

ARM_A = "ARM_A_OFFLINE_REPLAY"
ARM_B = "ARM_B_BUDGET_MATCHED_RECENT"
ARM_C = "ARM_C_NO_EXTRA_TRAINING"
ARM_D = "ARM_D_ADDITIVE_BUDGET"
ARMS = (ARM_A, ARM_B, ARM_C, ARM_D)

# Seed 44 is deliberately absent (recurring per-seed instability on reef-config
# envs, EXQ-539/540, V3-EXQ-538a); 45 is the sanctioned substitute.
SEEDS = (42, 123, 456, 2026, 45)

# ---------------------------------------------------------------------------
# Life geometry. Total buffered steps = TRAIN_EPISODES * EPISODE_STEPS = 420,
# comfortably under the 1000-entry FIFO cap on _world_experience_buffer /
# _e2_transition_buffer (agent.py) -- so the WHOLE life stays resident and
# precondition (ii) is structurally satisfiable rather than capped away.
# ---------------------------------------------------------------------------
GRID_SIZE = 9
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16
ALPHA = 0.9  # alpha_world / alpha_self; see docstring (0.3 default degenerates)

EPISODE_STEPS = 30
WARMUP_EPISODES = 6      # early regime, trains, no consolidation point
EARLY_EPISODES = 6       # early regime, trains, consolidation at 3 and 6
LATE_EPISODES = 6        # late  regime, trains, consolidation at 3 and 6
TRAIN_EPISODES = WARMUP_EPISODES + EARLY_EPISODES + LATE_EPISODES   # = 18 (the ep N/M denominator)
# Warmup and per-episode waking budget were raised (2 -> 6 episodes, 16 -> 24 steps)
# after a 3-seed full-scale measurement showed E1 sitting BELOW a do-nothing
# persistence predictor on some seeds -- an under-trained world model, which makes a
# "fidelity" reading hard to interpret even though the A-vs-B ordering was already
# 3/3 consistent. The lever is training strength, not a relaxed gate.
# Life length stays inside the 1000-entry buffer FIFO cap: 18 * 30 = 540.
PROBE_EPISODES = 4       # per stratum; HELD OUT (buffers truncated after capture)

CONSOLIDATE_EVERY = 3    # episodes, within the early and late blocks
RECENT_WINDOW = CONSOLIDATE_EVERY * EPISODE_STEPS   # = 90: arm B's visible slice
# NOTE: the LIVE window is always recomputed from the episode length actually in
# use (`_recent_window()`), never read off this constant. Holding it fixed while
# --dry-run shrinks the episode makes the window exceed the whole buffer, and arm
# B silently collapses onto arm A -- which is exactly what the first smoke of this
# script did (A and B bit-identical). The `replay_window_separation` and
# `manipulation_reaches_dv` preconditions below now fail that state outright.

CMC_STEPS = 24           # consolidation steps per point (A and B; C gets 0)
WAKE_STEPS = 24          # per-episode online steps (ALL arms, identical)
# Two learning rates, deliberately different and for different jobs.
# WAKE_LR drives the recency pressure the claim's premise needs (forgetting of
# early states must actually happen, or nothing is there for replay to counter).
# CMC_LR is the OFFLINE consolidation step, deliberately gentler: at 1e-3 the
# consolidation pass overshot so hard that BOTH arms A and B ended up WORSE than a
# do-nothing persistence predictor on the early probes (measured -0.36 to -0.77
# skill across three independent runs), which is the state the red-team's Finding 2
# calls uninterpretable. A and B share this value exactly -- it is not part of the
# contrast, and the gradient-budget match is enforced separately.
WAKE_LR = 1e-3
CMC_LR = 2e-4
CMC_BATCH = 16

# Two spatially separated regimes -> early and late states genuinely differ.
EARLY_START = (2, 2)
LATE_START = (6, 6)
HAZARDS = [(4, 4)]
RESOURCES = [(4, 2)]
# action indices: 0 up, 1 down, 2 left, 3 right (CausalGridWorldV2.ACTIONS)
EARLY_ACTION_P = (0.35, 0.15, 0.35, 0.15)   # up/left biased
LATE_ACTION_P = (0.15, 0.35, 0.15, 0.35)    # down/right biased

# ---------------------------------------------------------------------------
# PRE-REGISTERED thresholds (constants; never derived from this run's own stats)
# ---------------------------------------------------------------------------
SIGN_CONSISTENCY_REQUIRED = 4        # of len(SEEDS) = 5
LATE_TOL = 0.10                      # A may be up to 10% worse than B on LATE probes
MIN_REL_GAIN_EARLY = 0.05            # mean (B-A)/B on EARLY probes must clear 5%
FORGETTING_FLOOR = 1.10              # precondition (i): C_early / C_late
REPLAY_EARLY_SHARE_FLOOR = 0.20      # precondition (ii)
UPDATES_FLOOR = 1.0                  # precondition (iii)
STATE_DIVERGENCE_FLOOR = 0.20        # precondition (iv), Jaccard distance
PROBE_TARGET_VAR_FLOOR = 1e-4        # instrument gate: probe targets must move

MIN_WINDOW_SEPARATION = 0.20         # A's early-regime share minus B's
PERSISTENCE_SKILL_FLOOR = 0.0        # E1 must beat a do-nothing forward model

PROBE_WINDOW_STRIDE = 2   # denser windows -> lower DV variance
EPS = 1e-12

# `replay_window_separation`'s 0.20 floor is guaranteed by ARITHMETIC at design
# time, not hoped for -- which is what this exemption is for. At every LATE-block
# consolidation point arm B's visible slice is the last CONSOLIDATE_EVERY (3)
# episodes, all of which are late-regime by construction, so B's early-regime
# share is exactly 0.000. Arm A sees the whole buffer, whose early-regime prefix is
# (WARMUP_EPISODES + EARLY_EPISODES) * EPISODE_STEPS = 8 * 30 = 240 steps out of at
# most TRAIN_EPISODES * EPISODE_STEPS = 420, i.e. a share of at least 240/420 =
# 0.571 (and 240/330 = 0.727 at the first late point). Minimum separation is
# therefore >= 0.571, ~2.9x the floor, before any learning happens. The separate
# `manipulation_reaches_dv` precondition supplies the DV-side headroom check the
# same warning asks for: it fails the run when A and B produce a bit-identical DV.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "replay_window_separation's floor is established by construction: arm B's "
    "late-block window is all-late-regime (share 0.000) and arm A's whole-buffer "
    "share is >= 240/420 = 0.571, so separation >= 0.571 vs a 0.20 floor. DV-side "
    "headroom is covered by the manipulation_reaches_dv precondition."
)


def _recent_window(episode_steps: int) -> int:
    """Arm B's visible slice, in buffered steps, for the episode length in use."""
    return CONSOLIDATE_EVERY * int(episode_steps)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """IDENTICAL config in every arm -- the arms differ only in the driver's
    consolidation call, never in agent construction."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA,
        alpha_self=ALPHA,
    )
    return REEAgent(cfg)


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _tensor_digest(chunks: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in chunks:
        h.update(t.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class _BufferWindow:
    """Temporarily narrow the four replay buffers the consolidation losses read.

    Restores the ORIGINAL list objects on exit (the slices are new lists holding
    the same tensor references, so nothing is copied or lost). ``window=None``
    leaves everything untouched -- that is arm A's whole-life case, and it goes
    through this same object so A and B differ by the slice bound alone.
    """

    _NAMES = (
        "_self_experience_buffer",
        "_world_experience_buffer",
        "_action_experience_buffer",
        "_e2_transition_buffer",
    )

    def __init__(self, agent: REEAgent, window: Optional[int]):
        self.agent = agent
        self.window = window
        self._saved: Dict[str, list] = {}

    def __enter__(self) -> "_BufferWindow":
        if self.window is None:
            return self
        for name in self._NAMES:
            buf = getattr(self.agent, name)
            self._saved[name] = buf
            setattr(self.agent, name, buf[-self.window:])
        return self

    def __exit__(self, *exc) -> bool:
        for name, buf in self._saved.items():
            setattr(self.agent, name, buf)
        self._saved.clear()
        return False


def _consolidate(
    agent: REEAgent,
    consolidator: CrossModuleConsolidator,
    n_steps: int,
    window: Optional[int],
    lr: float,
) -> Dict[str, float]:
    """One live MECH-423 R3 consolidation pass over the windowed buffers."""
    with _BufferWindow(agent, window):
        return consolidator.consolidate(
            module_losses={
                "e1": lambda: agent.compute_prediction_loss(),
                "e2": lambda: agent.compute_e2_loss(batch_size=CMC_BATCH),
            },
            module_params={
                "e1": list(agent.e1.parameters()),
                "e2": list(agent.e2.parameters()),
            },
            n_steps=n_steps,
            schedule="interleaved",
            lr=lr,
            simulation_mode=False,
        )


def _early_share(regime_flags: List[int], window: Optional[int]) -> float:
    """Share of the buffer slice visible to the pass that is EARLY-regime.

    ``regime_flags`` is one entry per buffered step, 1 for the early regime
    (warmup + early block), 0 for the late block -- kept in lockstep with
    ``_world_experience_buffer`` by the training loop.
    """
    view = regime_flags if window is None else regime_flags[-window:]
    if not view:
        return 0.0
    return float(sum(view)) / float(len(view))


# ---------------------------------------------------------------------------
# probe capture + held-out readouts
# ---------------------------------------------------------------------------
def _e1_probe_mse(agent: REEAgent, episodes: List[Dict]) -> Tuple[float, float, float]:
    """Held-out MULTI-STEP E1 error, mirroring compute_prediction_loss exactly.

    Returns (mean MSE over windows, variance of the concatenated targets, mean
    PERSISTENCE-baseline MSE over the same windows). The persistence baseline is
    the do-nothing forward model -- predict the window's initial state for every
    horizon step -- and exists so a below-baseline E1 is caught as an instrument
    failure rather than read as a fidelity result.
    """
    horizon = int(agent.e1.config.prediction_horizon)
    action_cond = bool(getattr(agent.e1.config, "action_conditioned_transition", False))
    saved_hidden = agent.e1._hidden_state
    losses: List[float] = []
    persistence: List[float] = []
    target_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for ep in episodes:
            combined = ep["combined"]              # [1, T, total_dim]
            actions = ep["actions"]                # [1, T, action_dim] or None
            total = int(combined.shape[1])
            start = 0
            while start + 2 <= total:
                end = min(start + horizon + 1, total)
                if end - start < 2:
                    break
                seq = combined[:, start:end, :]
                agent.e1.reset_hidden_state()
                initial = seq[:, 0, :]
                horizon_len = int(seq.shape[1]) - 1
                acts = None
                if action_cond and actions is not None:
                    # Same +1 offset compute_prediction_loss uses: the action that
                    # carries state_i -> state_{i+1} is recorded alongside state_{i+1}.
                    acts = actions[:, start + 1:end, :]
                    if int(acts.shape[1]) != horizon_len:
                        acts = None
                preds = agent.e1.predict_long_horizon(
                    initial, horizon=horizon_len, actions=acts
                )
                targets = seq[:, 1:, :]
                losses.append(
                    float(F.mse_loss(preds[:, :targets.shape[1], :], targets).item())
                )
                persistence.append(float(F.mse_loss(
                    initial.unsqueeze(1).expand_as(targets), targets
                ).item()))
                target_chunks.append(targets.reshape(-1))
                if end >= total:
                    break
                start += PROBE_WINDOW_STRIDE
    agent.e1._hidden_state = saved_hidden
    if not losses:
        return float("nan"), 0.0, float("nan")
    var = float(torch.cat(target_chunks).var(unbiased=False).item()) if target_chunks else 0.0
    return _mean(losses), var, _mean(persistence)


def _e2_probe_mse(agent: REEAgent, episodes: List[Dict]) -> float:
    """Held-out ONE-STEP E2 SELF-forward error (predict_next_self).

    NOT `E2.world_forward` -- that head has no trainer anywhere in ree_core
    (substrate_queue `e2-world-forward-sleep-trainer`), so a world_forward DV
    would be identically 0.0 in every arm. Secondary / non-load-bearing.
    """
    zs, acts, zs1 = [], [], []
    for ep in episodes:
        for z_t, a_t, z_t1 in ep["transitions"]:
            zs.append(z_t)
            acts.append(a_t)
            zs1.append(z_t1)
    if not zs:
        return float("nan")
    with torch.no_grad():
        pred = agent.e2.predict_next_self(torch.cat(zs, dim=0), torch.cat(acts, dim=0))
        return float(F.mse_loss(pred, torch.cat(zs1, dim=0)).item())


# ---------------------------------------------------------------------------
# one (arm, seed) cell
# ---------------------------------------------------------------------------
def run_cell(
    arm: str,
    seed: int,
    warmup_eps: int,
    early_eps: int,
    late_eps: int,
    probe_eps: int,
    episode_steps: int,
    cmc_steps: int,
) -> Dict:
    print(f"Seed {seed} Condition {arm}", flush=True)

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device
    consolidator = CrossModuleConsolidator(
        CrossModuleConsolidatorConfig(schedule="interleaved", n_steps=cmc_steps, lr=CMC_LR)
        # lr is overridden per call by _consolidate(); this default is never used.
    )
    # Model-INDEPENDENT action stream: identical in every arm of this seed.
    rng = torch.Generator(device="cpu").manual_seed(seed)

    recent_window = _recent_window(episode_steps)
    train_total = warmup_eps + early_eps + late_eps
    regime_flags: List[int] = []          # 1 per buffered step: 1 = early regime
    occupancy = {"early": set(), "late": set()}
    cmc_records: List[Dict] = []
    ep_index = 0

    def _run_episode(regime: str, capture: bool) -> Optional[Dict]:
        """One 30-step episode. ``capture=True`` holds the episode OUT of the
        training buffers (they are truncated back afterwards) and returns its
        probe tensors instead."""
        nonlocal ep_index
        pre_lens = {
            name: len(getattr(agent, name))
            for name in _BufferWindow._NAMES
        }
        pre_flags = len(regime_flags)

        start_pos = EARLY_START if regime == "early" else LATE_START
        probs = torch.tensor(
            EARLY_ACTION_P if regime == "early" else LATE_ACTION_P, dtype=torch.float32
        )
        _, obs_dict = env.reset_to(start_pos, HAZARDS, RESOURCES)
        agent.e1.reset_hidden_state()

        prev_action: Optional[torch.Tensor] = None
        combined_rows: List[torch.Tensor] = []
        action_rows: List[torch.Tensor] = []
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for _step in range(episode_steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )
            # The action that CARRIED z_{t-1} -> z_t is the one sampled on the
            # PREVIOUS step, not the one about to be sampled now. Red-team minor
            # note (a): several landed harnesses key E2 on `act_prev`
            # (_lib/allon_training.py:395); keying on the not-yet-executed action
            # trains E2 on a mapping that never happened. Probe and training use
            # this same corrected convention, so the held-out E2 readout stays a
            # fair measure of what E2 was trained to do.
            caused_by = prev_action

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)
            if not capture:
                regime_flags.append(1 if regime == "early" else 0)

            occupancy[regime].add(tuple(env.get_agent_position()))

            action_idx = int(torch.multinomial(probs, 1, generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0

            if capture:
                # The probe transition triple uses the SAME (prev_z_self, action,
                # z_self) convention as the record_transition() call below, i.e.
                # exactly what compute_e2_loss trains on. Held-out probe and
                # training data must share the convention or the E2 readout is
                # not a fair held-out measure of what E2 was trained to do.
                combined_rows.append(
                    torch.cat(
                        [latent.z_self.detach().squeeze(0), latent.z_world.detach().squeeze(0)]
                    )
                )
                action_rows.append(agent._e1_action_one_hot().detach().reshape(1, -1))
                if prev_z_self is not None and caused_by is not None:
                    transitions.append(
                        (prev_z_self, caused_by.clone(), latent.z_self.detach().clone())
                    )
            if prev_z_self is not None and caused_by is not None:
                agent.record_transition(prev_z_self, caused_by, latent.z_self.detach())
            prev_action = action

            _, harm_signal, done, _info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset_to(start_pos, HAZARDS, RESOURCES)
                agent.e1.reset_hidden_state()

        if capture:
            # HOLD OUT: truncate every buffer back to its pre-episode length so
            # nothing from a probe episode can be trained on or replayed.
            for name, n in pre_lens.items():
                del getattr(agent, name)[n:]
            del regime_flags[pre_flags:]
            acts_t = torch.stack([a.reshape(-1) for a in action_rows]).unsqueeze(0).to(device)
            return {
                "combined": torch.stack(combined_rows).unsqueeze(0),
                "actions": acts_t,
                "transitions": transitions,
            }

        ep_index += 1
        print(f"  [train] {arm} seed={seed} ep {ep_index}/{train_total}", flush=True)
        # Waking training -- IDENTICAL in all arms: budgeted steps over THIS
        # episode's entries only (the online stream, no replay).
        _consolidate(agent, consolidator, WAKE_STEPS, window=episode_steps, lr=WAKE_LR)
        return None

    def _one_pass(block: str, label: str, window: Optional[int]) -> None:
        """One consolidation pass, recorded. `window=None` is the whole-life buffer."""
        share = _early_share(regime_flags, window)
        metrics = _consolidate(agent, consolidator, cmc_steps, window, lr=CMC_LR)
        cmc_records.append({
            "block": block,
            "pass": label,
            "skipped": False,
            "early_regime_share": share,
            "n_updates": float(metrics.get("n_updates", 0.0)),
            "updates_e1": float(metrics.get("updates_e1", 0.0)),
            "updates_e2": float(metrics.get("updates_e2", 0.0)),
            "cross_module_replay_share": float(metrics.get("cross_module_replay_share", 0.0)),
        })

    def _consolidation_point(block: str) -> None:
        """The manipulation. Arm C skips it entirely; arm D runs BOTH passes.

        ARM_D is the ADDITIVE-budget arm: at each point it takes ARM_A's
        whole-buffer pass AND ARM_B's recent-window pass, as two INDEPENDENT
        `consolidator.consolidate(...)` calls run SEQUENTIALLY -- never one
        accumulated backward. `_consolidate` wraps each call in its own
        `_BufferWindow`, so the whole-buffer pass has fully exited the context
        manager (buffers restored) before the recent-window pass narrows them;
        the second pass therefore cannot inherit the first's narrowing. Total
        extra budget per point is exactly 2 x cmc_steps = A's budget + B's.
        """
        if arm == ARM_C:
            cmc_records.append({"block": block, "pass": "none", "skipped": True,
                                "n_updates": 0.0})
            return
        if arm == ARM_D:
            _one_pass(block, "whole_buffer", None)          # ARM_A's pass
            _one_pass(block, "recent_window", recent_window)  # ARM_B's pass
            return
        _one_pass(block, "whole_buffer" if arm == ARM_A else "recent_window",
                  None if arm == ARM_A else recent_window)

    # ---- P0: warmup (early regime) ----------------------------------------
    for _ in range(warmup_eps):
        _run_episode("early", capture=False)

    # ---- EARLY block ------------------------------------------------------
    for i in range(early_eps):
        _run_episode("early", capture=False)
        if (i + 1) % CONSOLIDATE_EVERY == 0:
            _consolidation_point("early")

    # ---- EARLY probe capture (held out) -----------------------------------
    early_probes = [_run_episode("early", capture=True) for _ in range(probe_eps)]
    # Held-out EARLY error measured NOW, while the early states are still the
    # recent ones. Re-measured at the end of the life; the ratio of the two is a
    # WITHIN-STRATUM forgetting measure, which an early-vs-late comparison is not
    # (the two regimes sit in different parts of the grid, so a cross-stratum
    # ratio is confounded by intrinsic difficulty -- confirmed at full scale,
    # where ARM_A's early probes came out EASIER than its late ones).
    e1_early_at_capture, _, _ = _e1_probe_mse(agent, early_probes)

    # ---- LATE block -------------------------------------------------------
    for i in range(late_eps):
        _run_episode("late", capture=False)
        if (i + 1) % CONSOLIDATE_EVERY == 0:
            _consolidation_point("late")

    # ---- LATE probe capture (held out) ------------------------------------
    late_probes = [_run_episode("late", capture=True) for _ in range(probe_eps)]

    # ---- P2: held-out readouts -------------------------------------------
    e1_early, var_early, pers_early = _e1_probe_mse(agent, early_probes)
    e1_late, var_late, pers_late = _e1_probe_mse(agent, late_probes)
    skill_early = 1.0 - (e1_early / max(pers_early, EPS))
    skill_late = 1.0 - (e1_late / max(pers_late, EPS))
    forgetting_ratio = e1_early / max(e1_early_at_capture, EPS)
    e2_early = _e2_probe_mse(agent, early_probes)
    e2_late = _e2_probe_mse(agent, late_probes)

    late_points = [r for r in cmc_records if r["block"] == "late" and not r["skipped"]]
    replay_early_share = (
        _mean([r["early_regime_share"] for r in late_points]) if late_points else 0.0
    )
    # ARM_D runs TWO passes per point, so the cell-level mean above blends them.
    # Report the WHOLE-BUFFER pass separately -- that is the replay component, and
    # it is what makes D's coverage comparable with ARM_A's. Descriptive only: no
    # precondition or criterion reads it (replay_covers_early_regime is ARM_A-only,
    # replay_window_separation is ARM_A minus ARM_B; both unchanged).
    _wb_late = [r for r in late_points if r.get("pass") == "whole_buffer"]
    replay_early_share_whole_buffer = (
        _mean([r["early_regime_share"] for r in _wb_late]) if _wb_late else 0.0
    )
    window_label = {ARM_A: "ALL", ARM_B: str(recent_window), ARM_C: "none",
                    ARM_D: "ALL+" + str(recent_window)}[arm]
    updates_e1 = sum(float(r.get("updates_e1", 0.0)) for r in cmc_records)
    updates_e2 = sum(float(r.get("updates_e2", 0.0)) for r in cmc_records)
    total_extra_steps = sum(float(r.get("n_updates", 0.0)) for r in cmc_records)

    only_early = occupancy["early"] - occupancy["late"]
    only_late = occupancy["late"] - occupancy["early"]
    union = occupancy["early"] | occupancy["late"]
    jaccard_distance = (len(only_early) + len(only_late)) / float(len(union)) if union else 0.0

    probe_digest = _tensor_digest(
        [ep["combined"].reshape(-1) for ep in early_probes + late_probes]
    )
    writepath_mode = str(getattr(agent.e1.config, "sd016_writepath_mode", "off"))
    finite = all(
        torch.isfinite(p).all().item()
        for m in (agent.e1, agent.e2) for p in m.parameters()
    )

    print(
        f"  {arm} seed={seed} e1_early={e1_early:.6g} e1_late={e1_late:.6g} "
        f"e1_skill_early={skill_early:.4f} forget_ratio={forgetting_ratio:.3f} "
        f"probe_var_early={var_early:.4g} "
        f"e2_early={e2_early:.6g} e2_late={e2_late:.6g} "
        f"extra_steps={total_extra_steps:.0f} updates_e1={updates_e1:.0f} "
        f"updates_e2={updates_e2:.0f} window={window_label} "
        f"replay_early_share={replay_early_share:.3f} "
        f"replay_early_share_whole_buffer_pass={replay_early_share_whole_buffer:.3f} "
        f"jaccard_dist={jaccard_distance:.3f} finite={finite}",
        flush=True,
    )
    cell_ok = bool(finite and e1_early == e1_early and e1_late == e1_late)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "e1_holdout_mse_early": e1_early,
        "e1_holdout_mse_late": e1_late,
        "e2_selfforward_holdout_mse_early": e2_early,
        "e2_selfforward_holdout_mse_late": e2_late,
        "probe_target_variance_early": var_early,
        "probe_target_variance_late": var_late,
        "e1_persistence_baseline_mse_early": pers_early,
        "e1_persistence_baseline_mse_late": pers_late,
        "e1_skill_over_persistence_early": skill_early,
        "e1_skill_over_persistence_late": skill_late,
        "e1_holdout_mse_early_at_capture": e1_early_at_capture,
        "within_stratum_forgetting_ratio": forgetting_ratio,
        "recent_window_steps": recent_window,
        "total_extra_gradient_steps": total_extra_steps,
        "updates_e1": updates_e1,
        "updates_e2": updates_e2,
        "replay_early_regime_share": replay_early_share,
        "replay_early_regime_share_whole_buffer_pass": replay_early_share_whole_buffer,
        "n_consolidation_passes": float(len([r for r in cmc_records if not r["skipped"]])),
        "consolidation_points": cmc_records,
        "early_late_jaccard_distance": jaccard_distance,
        "n_cells_early_only": len(only_early),
        "n_cells_late_only": len(only_late),
        "n_cells_union": len(union),
        "probe_digest": probe_digest,
        "sd016_writepath_mode": writepath_mode,
        "params_finite": finite,
        "cell_ok": cell_ok,
        "agent": agent,
    }


# ---------------------------------------------------------------------------
def _flat_scalar(d: Dict[str, object]) -> Dict[str, float]:
    """Flat numeric readout: bools as 0/1 ints, non-finite DROPPED."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            f = float(v)
            if f == f and abs(f) != float("inf"):
                out[k] = f
    return out


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    seeds = (SEEDS[0],) if dry_run else SEEDS
    warmup_eps = 1 if dry_run else WARMUP_EPISODES
    early_eps = CONSOLIDATE_EVERY if dry_run else EARLY_EPISODES
    late_eps = CONSOLIDATE_EVERY if dry_run else LATE_EPISODES
    probe_eps = 1 if dry_run else PROBE_EPISODES
    episode_steps = 12 if dry_run else EPISODE_STEPS
    cmc_steps = 4 if dry_run else CMC_STEPS

    print(
        f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) arms={ARMS} seeds={seeds} "
        f"train_eps={warmup_eps + early_eps + late_eps} steps={episode_steps} "
        f"cmc_steps={cmc_steps} recent_window={_recent_window(episode_steps)}",
        flush=True,
    )
    t0 = time.time()

    rows: Dict[Tuple[str, int], Dict] = {}
    arm_results: List[Dict] = []
    agents_seen: List[REEAgent] = []
    for arm in ARMS:
        for seed in seeds:
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
                "self_dim": SELF_DIM,
                "world_dim": WORLD_DIM,
                "alpha": ALPHA,
                "episode_steps": episode_steps,
                "warmup_episodes": warmup_eps,
                "early_episodes": early_eps,
                "late_episodes": late_eps,
                "probe_episodes": probe_eps,
                "consolidate_every": CONSOLIDATE_EVERY,
                "recent_window": _recent_window(episode_steps),
                "cmc_steps": cmc_steps,
                "wake_steps": WAKE_STEPS,
                "cmc_lr": CMC_LR,
                "cmc_batch": CMC_BATCH,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
                row = run_cell(
                    arm, seed, warmup_eps, early_eps, late_eps, probe_eps,
                    episode_steps, cmc_steps,
                )
                agents_seen.append(row.pop("agent"))
                cell.stamp(row)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    a_rows = [rows[(ARM_A, s)] for s in seeds]
    b_rows = [rows[(ARM_B, s)] for s in seeds]
    c_rows = [rows[(ARM_C, s)] for s in seeds]
    d_rows = [rows[(ARM_D, s)] for s in seeds]
    all_rows = a_rows + b_rows + c_rows + d_rows

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    # ---------------- preconditions (the claim's own non-degeneracy list) ----
    forgetting_ratio = _mean([r["within_stratum_forgetting_ratio"] for r in c_rows])
    min_replay_share_a = min(r["replay_early_regime_share"] for r in a_rows)
    replay_share_b = _mean([r["replay_early_regime_share"] for r in b_rows])
    min_updates_e1_a = min(r["updates_e1"] for r in a_rows)
    min_updates_e2_a = min(r["updates_e2"] for r in a_rows)
    min_jaccard = min(r["early_late_jaccard_distance"] for r in all_rows)
    min_probe_var = min(
        min(r["probe_target_variance_early"], r["probe_target_variance_late"])
        for r in all_rows
    )
    n_seeds_probe_match = sum(
        1 for s in seeds
        if len({rows[(a, s)]["probe_digest"] for a in ARMS}) == 1
    )
    n_writepath_off = sum(
        1 for r in all_rows if r["sd016_writepath_mode"] == "off"
    )
    n_cells = len(ARMS) * len(seeds)
    budget_a = _mean([r["total_extra_gradient_steps"] for r in a_rows])
    budget_b = _mean([r["total_extra_gradient_steps"] for r in b_rows])
    # ARM_D is deliberately budget-UNMATCHED -- that IS the manipulation. Its budget
    # must be exactly A's plus B's (2 x CMC_STEPS per point), so a reader can verify
    # the budget was ADDED rather than reallocated. This is an instrument gate on the
    # new manipulation, not a pass/fail criterion for the question.
    budget_d = _mean([r["total_extra_gradient_steps"] for r in d_rows])
    additive_budget_error = abs(budget_d - (budget_a + budget_b))
    # The manipulation must actually SEPARATE the two replay windows, and must
    # actually REACH the DV. The first smoke of this script failed both (arm B's
    # window exceeded the whole buffer, so A and B were bit-identical) -- these
    # two gates turn that state into substrate_not_ready_requeue instead of a
    # spurious "A == B falsifies MECH-017".
    min_window_separation = min(
        rows[(ARM_A, s_)]["replay_early_regime_share"]
        - rows[(ARM_B, s_)]["replay_early_regime_share"]
        for s_ in seeds
    )
    n_seeds_dv_moved = sum(
        1 for s_ in seeds
        if rows[(ARM_A, s_)]["e1_holdout_mse_early"]
        != rows[(ARM_B, s_)]["e1_holdout_mse_early"]
    )
    # Red-team Finding 2 (CONFIRMED): the original gate read ARM_C alone -- an arm the
    # A-vs-B verdict never uses -- so a run could record `supports` for "fidelity beyond
    # matched compute" while NEITHER compared arm had any fidelity (measured: A -0.361,
    # B -0.653 on a draft where C was positive).
    #
    # The fix ranges over the COMPARED arms, and is deliberately a per-seed MAX, not a
    # min over every arm. Two states are legitimate results here and must NOT be gated
    # away as instrument failures:
    #   * ARM_C below persistence is the PHENOMENON (it is the no-extra-training arm;
    #     its early-state fidelity decaying is what `forgetting_present_in_control`
    #     measures). Full-scale seed 42: C = -0.297 while A = +0.409.
    #   * ARM_B below persistence is a STRONG confirmation, not a defect: replay kept a
    #     world model where budget-matched recency-only training did not.
    # What is genuinely uninterpretable is a seed where NEITHER compared arm beats a
    # do-nothing predictor -- then the A-vs-B difference is noise about noise. So the
    # gate is: every seed must carry at least one interpretable compared arm.
    per_seed_best_skill = {
        s_: max(rows[(ARM_A, s_)]["e1_skill_over_persistence_early"],
                rows[(ARM_B, s_)]["e1_skill_over_persistence_early"])
        for s_ in seeds
    }
    min_best_skill_ab = min(per_seed_best_skill.values())
    worst_skill_seed = min(per_seed_best_skill, key=per_seed_best_skill.get)
    n_seeds_additive_dv_moved = sum(
        1 for s_ in seeds
        if rows[(ARM_D, s_)]["e1_holdout_mse_late"]
        != rows[(ARM_B, s_)]["e1_holdout_mse_late"]
    )
    min_skill_ab = min(r["e1_skill_over_persistence_early"] for r in a_rows + b_rows)
    min_skill_c = min(r["e1_skill_over_persistence_early"] for r in c_rows)
    worst_forget_seed = min(
        seeds, key=lambda s_: rows[(ARM_C, s_)]["within_stratum_forgetting_ratio"]
    )

    try:
        preconditions = p0_readiness_gate([
            {"name": "forgetting_present_in_control", "measured": forgetting_ratio,
             "threshold": FORGETTING_FLOOR, "direction": "lower",
             "control": "MEAN over seeds, ARM_C only (no extra training): the WITHIN-STRATUM ratio "
                        "(EARLY-probe E1 MSE at end of life) / (same probe set, same arm, measured "
                        "immediately after capture, before the late block). Same probe tensors on "
                        "both sides, so intrinsic regime difficulty cancels -- a cross-stratum "
                        "early/late ratio would not. `met` is the same central-tendency comparison "
                        "the measured value reports. Below the floor the model has not forgotten "
                        "the early states, so there is nothing for replay to counter and all arms "
                        "tie. Read off an arm the A-vs-B verdict does not use.",
             "worst_seed": int(worst_forget_seed)},
            {"name": "replay_window_separation", "measured": min_window_separation,
             "threshold": MIN_WINDOW_SEPARATION, "direction": "lower",
             "control": "min over seeds of (ARM_A early-regime replay share - ARM_B early-regime "
                        "replay share). This is the MANIPULATION itself: if the two windows do not "
                        "separate, A and B are the same run and 'A == B' says nothing about "
                        "MECH-017. Caught exactly this in the first smoke of this script."},
            {"name": "manipulation_reaches_dv", "measured": float(n_seeds_dv_moved),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "seeds where ARM_A and ARM_B produced DIFFERENT early-probe E1 MSE. A "
                        "bit-identical pair is a dead manipulation, not a null result."},
            {"name": "e1_has_skill_over_persistence_compared_arms",
             "measured": min_best_skill_ab,
             "threshold": PERSISTENCE_SKILL_FLOOR, "direction": "lower", "comparator": ">",
             "control": "WORST SEED of max(ARM_A skill, ARM_B skill), where skill = 1 - (E1 "
                        "early-probe MSE / do-nothing persistence-baseline MSE on the same "
                        "windows). Ranges over the two arms C1/C3 actually compare -- NOT the "
                        "ARM_C reference, whose sub-persistence skill is the phenomenon under "
                        "study. A seed where NEITHER compared arm beats a do-nothing predictor "
                        "makes that seed's A-vs-B difference noise about noise.",
             "offending_seed": int(worst_skill_seed),
             "min_skill_over_compared_arms": min_skill_ab,
             "min_skill_arm_c_reference_only": min_skill_c},
            {"name": "gradient_budget_matched", "measured": abs(budget_a - budget_b),
             "threshold": 0.5, "direction": "upper",
             "control": "|extra gradient steps ARM_A - ARM_B|. This is the DESIGN INVARIANT that "
                        "makes C1/C3 a replay result rather than a compute result, so a mismatch "
                        "is an INSTRUMENT failure, not evidence about MECH-017. Red-team Finding 1 "
                        "(CONFIRMED): as a pass/fail CRITERION it routed a budget mismatch to "
                        "`weakens`; as a precondition it correctly routes to "
                        "substrate_not_ready_requeue."},
            {"name": "additive_budget_is_a_plus_b", "measured": additive_budget_error,
             "threshold": 0.5, "direction": "upper",
             "control": "|extra gradient steps ARM_D - (ARM_A + ARM_B)|. ARM_D is deliberately "
                        "budget-UNMATCHED against A and B -- that IS this run's manipulation -- so "
                        "the gradient_budget_matched gate above deliberately still compares A "
                        "against B ONLY. This gate is its counterpart for D: it asserts the budget "
                        "was genuinely ADDED (A's whole-buffer pass PLUS B's recent-window pass, "
                        "2 x CMC_STEPS per point) rather than reallocated. If D's budget is not "
                        "exactly A+B the dose of the manipulation is not what was pre-registered, "
                        "which is an INSTRUMENT failure, not evidence about the reallocation "
                        f"question. Measured: D={budget_d:.0f}, A={budget_a:.0f}, B={budget_b:.0f}."},
            {"name": "additive_manipulation_reaches_dv",
             "measured": float(n_seeds_additive_dv_moved),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "seeds where ARM_D and ARM_B produced DIFFERENT LATE-probe E1 MSE -- the "
                        "exact leg the load-bearing C4 reads. Same form and same threshold as "
                        "manipulation_reaches_dv above, re-pointed at the compared pair: a "
                        "bit-identical D/B pair is a dead manipulation, and would otherwise be "
                        "scored as a clean C4 PASS ('D is not worse than B') on no manipulation "
                        "at all."},
            {"name": "replay_covers_early_regime", "measured": min_replay_share_a,
             "threshold": REPLAY_EARLY_SHARE_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of the mean early-regime share of the buffer slice "
                        "visible at the LATE-block consolidation points. compute_prediction_loss "
                        "draws start_idx ~ Uniform[0, buf_len-1), so this share IS the per-draw "
                        f"probability of an early-regime trace. ARM_B comparator = {replay_share_b:.4f}.",
             "arm_b_share": replay_share_b},
            {"name": "consolidation_updated_e1", "measured": min_updates_e1_a,
             "threshold": UPDATES_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of CrossModuleConsolidator's own updates_e1 counter -- "
                        "the pass must actually move the module the load-bearing DV reads."},
            {"name": "consolidation_updated_e2", "measured": min_updates_e2_a,
             "threshold": UPDATES_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of updates_e2 (E2 SELF-forward; the secondary readout)."},
            {"name": "early_late_state_divergence", "measured": min_jaccard,
             "threshold": STATE_DIVERGENCE_FLOOR, "direction": "lower",
             "control": "min over ALL cells of the Jaccard DISTANCE between early-regime and "
                        "late-regime occupied grid cells. If old and recent states coincide there "
                        "is nothing for replay to add (sleep_substrate:GAP-2 vacuous case)."},
            {"name": "probe_set_identical_across_arms", "measured": float(n_seeds_probe_match),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "sha256 over each cell's concatenated probe [z_self, z_world] tensors; all "
                        "four arms of a seed must agree EXACTLY, which is what makes the probe set "
                        "FIXED rather than arm-dependent."},
            {"name": "probe_target_variance", "measured": min_probe_var,
             "threshold": PROBE_TARGET_VAR_FLOOR, "direction": "lower",
             "control": "min over all cells/strata of the variance of the held-out E1 target tensor -- "
                        "a frozen target would make the MSE trivially satisfiable."},
            {"name": "sd016_writepath_mode_off", "measured": float(n_writepath_off),
             "threshold": float(n_cells), "direction": "lower",
             "control": "runtime read of E1Config.sd016_writepath_mode in every cell. 'off' means the "
                        "open CORRUPTING defect contextmemory-write-path-addressing-degeneracy "
                        "(e1_deep.py::ContextMemory.write) is never called on this run's path."},
        ])
        gate_ok = True
    except P0NotReady as e:
        preconditions = e.preconditions
        gate_ok = False

    cells_ok = all(r["cell_ok"] for r in a_rows + b_rows + c_rows)

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "N/A -- no SleepLoopManager built; the MECH-423 R3 CrossModuleConsolidator "
            "is called directly (V3-EXQ-680e call shape). Sleep-cycle wiring of this "
            "consolidator is separately validated by V3-EXQ-1026."
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "substrate_limits_declared": {
            "e2_world_forward_readout": (
                "NOT ATTEMPTED. substrate_queue 'e2-world-forward-sleep-trainer' "
                "(pending_implementation, degrading): CrossModuleConsolidator updates only E2's "
                "SELF-forward head and E2.world_forward has no trainer in ree_core, so a "
                "world_forward DV is identically 0.0 in every arm. The E2 numbers reported here "
                "are SELF-forward and are secondary / non-load-bearing."
            ),
            "contextmemory_write_defect": (
                "OFF-PATH. sd016_writepath_mode='off' in every cell (asserted as a precondition), "
                "so e1_deep.py::ContextMemory.write is never called."
            ),
        },
        "config": {
            "arms": list(ARMS),
            "seeds": list(seeds),
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "alpha_world": ALPHA,
            "alpha_self": ALPHA,
            "episode_steps": episode_steps,
            "warmup_episodes": warmup_eps,
            "early_episodes": early_eps,
            "late_episodes": late_eps,
            "probe_episodes": probe_eps,
            "train_episodes": warmup_eps + early_eps + late_eps,
            "consolidate_every": CONSOLIDATE_EVERY,
            "recent_window": _recent_window(episode_steps),
            "cmc_steps": cmc_steps,
            "wake_steps": WAKE_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
            "early_start": list(EARLY_START),
            "late_start": list(LATE_START),
            "hazards": [list(h) for h in HAZARDS],
            "resources": [list(r) for r in RESOURCES],
            "registered_thresholds": {
                "min_window_separation": MIN_WINDOW_SEPARATION,
                "persistence_skill_floor": PERSISTENCE_SKILL_FLOOR,
                "wake_lr": WAKE_LR,
                "cmc_lr": CMC_LR,
                "sign_consistency_required": SIGN_CONSISTENCY_REQUIRED,
                "late_tol": LATE_TOL,
                "min_rel_gain_early": MIN_REL_GAIN_EARLY,
                "forgetting_floor": FORGETTING_FLOOR,
                "replay_early_share_floor": REPLAY_EARLY_SHARE_FLOOR,
                "updates_floor": UPDATES_FLOOR,
                "state_divergence_floor": STATE_DIVERGENCE_FLOOR,
                "probe_target_var_floor": PROBE_TARGET_VAR_FLOOR,
            },
        },
        "elapsed_seconds": elapsed,
    }

    def _write(manifest: Dict) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest,
            dry_run=False,
            config=manifest.get("config"),
            seeds=list(seeds),
            script_path=Path(__file__),
            agent=agents_seen,
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    # ---------------- FORK: any precondition unmet -> not a MECH-017 verdict --
    if not gate_ok or not cells_ok:
        unmet = [p["name"] for p in preconditions if not p.get("met", True)]
        if not cells_ok:
            unmet = unmet + ["cell_parameters_finite"]
        reason = "unmet preconditions: " + ", ".join(unmet)
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Non-degeneracy precondition unmet; this run does NOT bear on MECH-017 in "
                "either direction. " + reason
            ),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": preconditions,
                "criteria_non_degenerate": {},
            },
            "readout": _flat_scalar({
                "gate_ok": False,
                "forgetting_ratio_control": forgetting_ratio,
                "min_replay_early_share_a": min_replay_share_a,
                "replay_early_share_b": replay_share_b,
                "min_updates_e1_a": min_updates_e1_a,
                "min_updates_e2_a": min_updates_e2_a,
                "min_early_late_jaccard_distance": min_jaccard,
                "n_seeds_probe_set_identical": float(n_seeds_probe_match),
                "min_probe_target_variance": min_probe_var,
                "min_replay_window_separation": min_window_separation,
                "n_seeds_dv_moved": float(n_seeds_dv_moved),
                "min_per_seed_best_skill_ab": min_best_skill_ab,
                "min_skill_ab": min_skill_ab,
                "min_skill_c": min_skill_c,
            }),
        })
        return "FAIL", _write(manifest)

    # ---------------- load-bearing criteria ---------------------------------
    early_a = [r["e1_holdout_mse_early"] for r in a_rows]
    early_b = [r["e1_holdout_mse_early"] for r in b_rows]
    early_c = [r["e1_holdout_mse_early"] for r in c_rows]
    late_a = [r["e1_holdout_mse_late"] for r in a_rows]
    late_b = [r["e1_holdout_mse_late"] for r in b_rows]
    early_d = [r["e1_holdout_mse_early"] for r in d_rows]
    late_d = [r["e1_holdout_mse_late"] for r in d_rows]

    n_a_better_early = sum(1 for x, y in zip(early_a, early_b) if x < y)
    n_a_not_worse_late = sum(
        1 for x, y in zip(late_a, late_b) if x <= y * (1.0 + LATE_TOL)
    )
    rel_gain_early = _mean([
        (y - x) / max(y, EPS) for x, y in zip(early_a, early_b)
    ])
    c1 = bool(n_a_better_early >= SIGN_CONSISTENCY_REQUIRED)
    c2 = bool(n_a_not_worse_late >= SIGN_CONSISTENCY_REQUIRED)
    c3 = bool(rel_gain_early > MIN_REL_GAIN_EARLY)

    # THIS RUN'S QUESTION. Transcribed from C2 with D substituted for A -- same
    # DV (e1_holdout_mse_late), same comparator arm (B), same LATE_TOL = 0.10, same
    # SIGN_CONSISTENCY_REQUIRED. Nothing new was invented: the only edit is the arm.
    n_d_not_worse_late = sum(
        1 for x, y in zip(late_d, late_b) if x <= y * (1.0 + LATE_TOL)
    )
    c4 = bool(n_d_not_worse_late >= SIGN_CONSISTENCY_REQUIRED)
    rel_cost_late_d = _mean([(x - y) / max(y, EPS) for x, y in zip(late_d, late_b)])
    rel_cost_late_a = _mean([(x - y) / max(y, EPS) for x, y in zip(late_a, late_b)])

    criteria = [
        {"name": "C4_additive_budget_no_cost_on_late_probes", "load_bearing": True,
         "passed": c4, "measured": float(n_d_not_worse_late),
         "threshold": float(SIGN_CONSISTENCY_REQUIRED), "comparator": ">=",
         "detail": f"seeds with e1_holdout_mse_late(D) <= (1+{LATE_TOL})*e1_holdout_mse_late(B). "
                   "THE question of this run: PASS = the recency deficit V3-EXQ-1048 measured was "
                   "an artefact of REALLOCATING a fixed budget (it disappears once the replay pass "
                   "is ADDED to the online budget instead of taking its place); FAIL = the deficit "
                   "is INTRINSIC to replay (it survives at additive budget). Identical in form to "
                   "V3-EXQ-1048's C2 -- same DV, same comparator arm B, same LATE_TOL, same "
                   "seed-majority constant -- with ARM_D substituted for ARM_A."},
        {"name": "C1_replay_beats_budget_matched_on_early_probes", "load_bearing": False,
         "passed": c1, "measured": float(n_a_better_early),
         "threshold": float(SIGN_CONSISTENCY_REQUIRED), "comparator": ">=",
         "detail": "REPLICATION of V3-EXQ-1048 C1 (A vs B at matched budget), NOT load-bearing "
                   "here: seeds with e1_holdout_mse_early(A) < e1_holdout_mse_early(B). Carried "
                   "because the A and B arms are re-run unchanged and the comparison is free; it "
                   "is a corroboration channel for 1048, not this run's verdict."},
        {"name": "C2_no_cost_on_late_probes", "load_bearing": False,
         "passed": c2, "measured": float(n_a_not_worse_late),
         "threshold": float(SIGN_CONSISTENCY_REQUIRED), "comparator": ">=",
         "detail": f"REPLICATION of V3-EXQ-1048 C2, NOT load-bearing here: seeds with "
                   f"e1_holdout_mse_late(A) <= (1+{LATE_TOL})*e1_holdout_mse_late(B). It FAILED "
                   "5/5 on 1048; it is retained as the matched-budget reference leg that C4 is "
                   "asked against, and re-pointing it at ARM_C is explicitly forbidden (the claim "
                   "names arm B on this leg; 1048's autopsy learning_extracted[1])."},
        {"name": "C3_early_effect_size_clears_floor", "load_bearing": False,
         "passed": c3, "measured": rel_gain_early, "threshold": MIN_REL_GAIN_EARLY,
         "comparator": ">",
         "detail": "REPLICATION of V3-EXQ-1048 C3, NOT load-bearing here: mean over seeds of "
                   "(early_B - early_A)/early_B."},
    ]
    criteria_non_degenerate = {
        "C4_additive_budget_no_cost_on_late_probes": bool(
            len({round(x, 12) for x in late_d + late_b}) > 1
        ),
        "C1_replay_beats_budget_matched_on_early_probes": bool(
            len({round(x, 12) for x in early_a + early_b}) > 1
        ),
        "C2_no_cost_on_late_probes": bool(
            len({round(x, 12) for x in late_a + late_b}) > 1
        ),
        "C3_early_effect_size_clears_floor": bool(
            len({round(x, 12) for x in early_a + early_b}) > 1
        ),
    }

    degeneracy = check_degeneracy({
        "e1_holdout_mse_early": {"groups": [[a, b] for a, b in zip(early_a, early_b)]},
        "e1_holdout_mse_late": {"groups": [[a, b] for a, b in zip(late_a, late_b)]},
        "e1_holdout_mse_late_additive": {
            "groups": [[d, b] for d, b in zip(late_d, late_b)]},
    })

    all_pass = all(c["passed"] for c in criteria if c["load_bearing"])
    n_a_worse_early = sum(1 for x, y in zip(early_a, early_b) if x > y)
    # ONE load-bearing criterion, so the verdict grid is a clean two-way split. The
    # A-vs-B legs above are replication diagnostics and deliberately cannot move it:
    # C2 already failed 5/5 on V3-EXQ-1048, so leaving it load-bearing would pin this
    # run to FAIL no matter what the reallocation question answers.
    if all_pass:
        label = "recency_cost_is_reallocation_artefact_not_intrinsic"
        direction = "supports"
    else:
        label = "recency_cost_intrinsic_to_replay_survives_additive_budget"
        direction = "weakens"

    note = (
        f"MECH-017 RECENCY-COST conjunct ONLY, asked at ADDITIVE rather than reallocated "
        f"gradient budget. ARM_D takes ARM_A's whole-buffer consolidation pass AND ARM_B's "
        f"recent-window pass at every consolidation point, as two sequential independent "
        f"consolidate() calls (D={budget_d:.0f} extra steps vs A={budget_a:.0f}, "
        f"B={budget_b:.0f}; |D-(A+B)|={additive_budget_error:.1f}). LATE-probe held-out E1 MSE: "
        f"D within (1+{LATE_TOL}) of B on {n_d_not_worse_late}/{len(seeds)} seeds "
        f"(threshold {SIGN_CONSISTENCY_REQUIRED}); mean relative late cost D vs B "
        f"{rel_cost_late_d:+.4f} against A vs B {rel_cost_late_a:+.4f} on the same run. "
        f"SCOPE: this run answers the REALLOCATION question only. V3-EXQ-1048's confirmed "
        f"autopsy records that MECH-017 'is two claims wearing one id' ('replay counters "
        f"forgetting' SUPPORTED; 'at no cost to recency' WEAKENED) and flagged it for a "
        f"granularity review -- this result must NOT be read as adjudicating that bundle. "
        f"The matched-budget legs C1/C2/C3 are re-run unchanged as NON-load-bearing "
        f"replication of 1048 (C1 {n_a_better_early}/{len(seeds)}, C2 "
        f"{n_a_not_worse_late}/{len(seeds)}, C3 rel gain {rel_gain_early:.4f}). "
        f"Control (ARM_C, no extra training) forgetting ratio early/late = "
        f"{forgetting_ratio:.3f}. E2 numbers in this manifest are SELF-forward, NOT the "
        f"world_forward readout MECH-017 names -- that head has no trainer in ree_core "
        f"(substrate_queue e2-world-forward-sleep-trainer) and would be 0.0 in every arm."
    )

    print(f"\n[{EXPERIMENT_TYPE}] verdict:")
    for c in criteria:
        print(f"  {c['name']}: passed={c['passed']} measured={c['measured']:.6g} "
              f"threshold={c['threshold']}")
    print(f"  -> {label} ({'PASS' if all_pass else 'FAIL'}); elapsed={elapsed:.1f}s")

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": "PASS" if all_pass else "FAIL",
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_note": note,
        "interpretation": {
            "label": label,
            "criteria": criteria,
            "combination_rule": "PASS iff the single LOAD-BEARING criterion C4 passes (plain all() over the load_bearing subset, no any()/OR branching). C1/C2/C3 are retained with load_bearing:false as replication of V3-EXQ-1048's matched-budget legs and CANNOT move the verdict -- C2 failed 5/5 there, so leaving it load-bearing would pin this run to FAIL independently of the question it was queued to answer. The A-vs-B gradient-budget match, and the separate assertion that ARM_D's budget is exactly A+B, are both PRECONDITIONS, so a mismatch routes to substrate_not_ready_requeue instead of to a MECH-017 verdict.",
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": preconditions,
        },
        "readout": _flat_scalar({
            "n_seeds_a_better_early": float(n_a_better_early),
            "n_seeds_a_not_worse_late": float(n_a_not_worse_late),
            "n_seeds_a_worse_early": float(n_a_worse_early),
            "rel_gain_early_a_over_b": rel_gain_early,
            "mean_e1_holdout_mse_early_a": _mean(early_a),
            "mean_e1_holdout_mse_early_b": _mean(early_b),
            "mean_e1_holdout_mse_early_c": _mean(early_c),
            "mean_e1_holdout_mse_late_a": _mean(late_a),
            "mean_e1_holdout_mse_late_b": _mean(late_b),
            "mean_e1_holdout_mse_late_c": _mean([r["e1_holdout_mse_late"] for r in c_rows]),
            "mean_e2_selfforward_holdout_mse_early_a": _mean(
                [r["e2_selfforward_holdout_mse_early"] for r in a_rows]),
            "mean_e2_selfforward_holdout_mse_early_b": _mean(
                [r["e2_selfforward_holdout_mse_early"] for r in b_rows]),
            "forgetting_ratio_control": forgetting_ratio,
            "mean_within_stratum_forgetting_ratio_a": _mean(
                [r["within_stratum_forgetting_ratio"] for r in a_rows]),
            "mean_within_stratum_forgetting_ratio_b": _mean(
                [r["within_stratum_forgetting_ratio"] for r in b_rows]),
            "min_replay_early_share_a": min_replay_share_a,
            "replay_early_share_b": replay_share_b,
            "extra_gradient_steps_a": budget_a,
            "extra_gradient_steps_b": budget_b,
            "extra_gradient_steps_d": budget_d,
            "additive_budget_error_d_minus_a_plus_b": additive_budget_error,
            "n_seeds_d_not_worse_late": float(n_d_not_worse_late),
            "n_seeds_additive_dv_moved": float(n_seeds_additive_dv_moved),
            "rel_cost_late_d_over_b": rel_cost_late_d,
            "rel_cost_late_a_over_b": rel_cost_late_a,
            "mean_e1_holdout_mse_early_d": _mean(early_d),
            "mean_e1_holdout_mse_late_d": _mean(late_d),
            "mean_e2_selfforward_holdout_mse_late_d": _mean(
                [r["e2_selfforward_holdout_mse_late"] for r in d_rows]),
            "mean_e2_selfforward_holdout_mse_late_a": _mean(
                [r["e2_selfforward_holdout_mse_late"] for r in a_rows]),
            "mean_e2_selfforward_holdout_mse_late_b": _mean(
                [r["e2_selfforward_holdout_mse_late"] for r in b_rows]),
            "mean_replay_early_share_whole_buffer_pass_d": _mean(
                [r["replay_early_regime_share_whole_buffer_pass"] for r in d_rows]),
            "c4_pass": c4,
            "min_early_late_jaccard_distance": min_jaccard,
            "min_updates_e1_a": min_updates_e1_a,
            "min_updates_e2_a": min_updates_e2_a,
            "n_seeds_probe_set_identical": float(n_seeds_probe_match),
            "min_replay_window_separation": min_window_separation,
            "n_seeds_dv_moved": float(n_seeds_dv_moved),
            "min_per_seed_best_skill_ab": min_best_skill_ab,
            "min_skill_ab": min_skill_ab,
            "min_skill_c": min_skill_c,
            "mean_e1_skill_over_persistence_a": _mean(
                [r["e1_skill_over_persistence_early"] for r in a_rows]),
            "mean_e1_skill_over_persistence_b": _mean(
                [r["e1_skill_over_persistence_early"] for r in b_rows]),
            "c1_pass": c1, "c2_pass": c2, "c3_pass": c3,  # replication only
            "gradient_budget_delta_a_minus_b": abs(budget_a - budget_b),
            "min_per_seed_best_skill_ab": min_best_skill_ab,
            "overall_pass": all_pass,
        }),
        "per_seed_results": [
            {
                "seed": s,
                "e1_early_a": rows[(ARM_A, s)]["e1_holdout_mse_early"],
                "e1_early_b": rows[(ARM_B, s)]["e1_holdout_mse_early"],
                "e1_early_c": rows[(ARM_C, s)]["e1_holdout_mse_early"],
                "e1_late_a": rows[(ARM_A, s)]["e1_holdout_mse_late"],
                "e1_late_b": rows[(ARM_B, s)]["e1_holdout_mse_late"],
                "e1_late_c": rows[(ARM_C, s)]["e1_holdout_mse_late"],
                "e1_early_d": rows[(ARM_D, s)]["e1_holdout_mse_early"],
                "e1_late_d": rows[(ARM_D, s)]["e1_holdout_mse_late"],
                "e2_late_d": rows[(ARM_D, s)]["e2_selfforward_holdout_mse_late"],
                "e2_late_b": rows[(ARM_B, s)]["e2_selfforward_holdout_mse_late"],
                "extra_steps_d": rows[(ARM_D, s)]["total_extra_gradient_steps"],
                "e2_early_a": rows[(ARM_A, s)]["e2_selfforward_holdout_mse_early"],
                "e2_early_b": rows[(ARM_B, s)]["e2_selfforward_holdout_mse_early"],
                "e2_early_c": rows[(ARM_C, s)]["e2_selfforward_holdout_mse_early"],
                "probe_digests": {a: rows[(a, s)]["probe_digest"] for a in ARMS},
            }
            for s in seeds
        ],
    })
    manifest.update(degeneracy)
    return manifest["outcome"], _write(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
