#!/opt/local/bin/python3
"""
V3-EXQ-956 -- ContextMemory write-address selection: real-agent validation of
the THIRD mechanism, `contextmemory_write_selection="gumbel_learned"`,
against substrate_queue `contextmemory-write-path-addressing-degeneracy`
(severity: corrupting).

Chip: chip-20260827-queueexp-contextmemory-gumbel-learned-validation

experiment_purpose: diagnostic

WHY THIS RUN EXISTS AND HOW IT DIFFERS FROM V3-EXQ-943. V3-EXQ-943 (2026-08-20)
validated the first two write-address mechanisms (conscience bias, refractory
eligibility mask) under this same real-agent harness -- both are deterministic
transformations of an UNTRAINED score, so a flag flip alone was sufficient to
evaluate them. `gumbel_learned` (built 2026-08-27, chip-20260826-
contextmemory-gumbel-writeselect-build) is different in kind: it ships with a
real, VERIFIED gradient path (`ContextMemory.compute_write_addressing_loss`,
wired into `REEAgent.compute_prediction_loss()` via the new top-level
`REEConfig.contextmemory_write_addressing_loss_weight`, gated on weight > 0
AND write_selection == "gumbel_learned"), and the whole open question this
mechanism exists to answer is CONTENT-DISCRIMINATION: does training that
tagger actually make write-address selection content-conditioned, or does it
stay content-blind like the write path's rejected `gumbel` draft?

Per the architecture doc (REE_assembly/docs/architecture/
contextmemory_write_address_selection.md, "THIRD mechanism" section) and
ree-v3/CLAUDE.md's gumbel_learned entry: a driver that only sets
write_selection="gumbel_learned" with the loss weight left at its 0.0 default
exercises ONLY the annealed-Gumbel-noise occupancy effect (already proven
decisive at the CONTEXTMEMORY-UNIT level, 16/16 slots on 5/5 seeds,
independent of training -- see test_contextmemory_write_gumbel_learned.py)
and says NOTHING about content-discrimination. This experiment is the first
to run `contextmemory_write_addressing_loss_weight > 0` through a REAL
training schedule under a REAL agent (the 436e/436f / V3-EXQ-943 harness:
CausalGridWorldV2 + REEAgent, 100 episodes x 150 steps = 15,000 steps,
alternating safe/dangerous context every 5 episodes) -- MATCHING (not merely
the same order of magnitude as) V3-EXQ-907's own real schedule (100 episodes,
15,000 steps) that WAS sufficient to move `compute_diversification_loss()`
off its own symmetry-breaking saddle, unlike the toy 300-step SGD script the
architecture doc reports as inconclusive for THIS loss. (V3-EXQ-943's own
200-episode schedule is not used here: that choice was calibrated to an
occupancy-only question with no gradient dependency, and this driver's
correctness fix -- see "DELIBERATE DIVERGENCE" in _run_episode's docstring
below -- makes each step substantially more expensive than 943's, since a
real agent._e1_tick() forward pass now runs every step instead of never
running at all. 100 episodes stays a real, non-toy schedule by the 907
precedent while keeping the run tractable.)

ARMS (3, all seeds x all arms -- no separate "conditions" dimension):
  LEGACY            write_selection="argmin"                         (bit-
                     identical to the pre-fix path; reference/floor, NOT
                     gating -- matches V3-EXQ-943's own LEGACY arm exactly)
  GUMBEL_UNTRAINED  write_selection="gumbel_learned",
                     contextmemory_write_addressing_loss_weight=0.0    (the
                     mechanism with the training signal OFF -- write_addr_
                     tagger stays at its random init for the whole run: the
                     addressing loss is gated on weight>0 in agent.py, so at
                     weight=0.0 it is never added to the backward graph and
                     the tagger's parameters receive literally zero gradient,
                     verified below via a before/after state_dict diff)
  GUMBEL_TRAINED    write_selection="gumbel_learned",
                     contextmemory_write_addressing_loss_weight=0.5    (training
                     signal ON. 0.5 is not an arbitrary choice: it mirrors
                     `sd016_diversification_weight`'s own validated operating
                     point -- LAMBDA_DIVERSIFY=0.5 is the value used
                     throughout the 418/436/907 lineage, see ree-v3/CLAUDE.md
                     and v3_exq_907_sd016_h1_ctxdiv.py -- which is the closest
                     precedent available since `compute_write_addressing_loss`
                     is architecturally the SAME mean-squared-off-diagonal-
                     cosine-similarity form applied to a complementary object
                     (per-example write-address selection distributions
                     instead of memory rows); no other value has been
                     measured for this specific loss, so 0.5 is the most
                     defensible starting point rather than an unjustified
                     pick.)
A 4th "BIAS"/"REFRACTORY" arm is deliberately NOT re-run here -- V3-EXQ-943
already validated those two mechanisms; this experiment is scoped to the
THIRD mechanism only, per this queue entry's own chip.

SEEDS: [42, 7, 13, 100, 200] -- identical to V3-EXQ-943, for read-across
comparability of the LEGACY reference arm and the training schedule.

TWO SEPARATE MEASUREMENT FAMILIES, both required by the acceptance criteria
below -- occupancy alone is NOT sufficient evidence this mechanism works,
because it is already proven at the unit level to hold even with an UNTRAINED
tagger (Gumbel noise alone guarantees it):

  1. OCCUPANCY (registered, expected to pass trivially per the architecture
     doc). Measured from REAL TRAINING DYNAMICS (train-mode writes, exactly
     V3-EXQ-943's WriteSequenceTracker reading ContextMemory.last_write_index
     / .slot_write_counts -- never re-deriving the selection expression, per
     the architecture doc's "Instrumentation change" mandate). NOT powered on
     occupied-slot cosine (occ_cos) or the DURING-training safe/dangerous
     context split, matching V3-EXQ-943's own prohibition
     (contextmemory_write_selection_comparison_20260819.md).

  2. CONTENT-DISCRIMINATION (the actually open question). Measured with the
     SAME 2-cluster synthetic content-conditioning instrument this queue
     entry's own contract tests use (test_contextmemory_write_address_
     selection.py's `_stream`/`_jaccard`, LATENT_DIM=64 -- which is exactly
     this driver's self_dim(32)+world_dim(32), so the instrument transfers
     without modification): AFTER real training completes, each cell's
     REAL, POSSIBLY-TRAINED `agent.e1.context_memory` is switched to
     agent.eval() (deterministic, no-RNG selection -- exactly plain argmin on
     write_addr_tagger's own scores per the architecture doc, isolating what
     the tagger has learned from Gumbel-noise contamination) and fed a fresh
     2-cluster synthetic stream (n=1500, same jitter=0.0078 the contract uses)
     directly via context_memory.write(state). Per-cluster occupied-slot sets
     give a Jaccard exactly as in the contract test.

     CORRECTION TO AN EARLIER ASSUMPTION, recorded here rather than silently
     fixed, per this file's own standard: an earlier draft of this docstring
     assumed the GUMBEL_UNTRAINED baseline would read ~1.000 (content-blind),
     "matching the unit-level measurement." No such unit-level measurement
     actually exists -- the contract file's own 1.000-on-5/5 numbers are
     either the REJECTED "gumbel" mode (a different mechanism entirely, see
     ree-v3/CLAUDE.md) or the FIRST-ATTEMPT loss design AFTER 300 SGD steps
     (a TRAINED, not untrained, tagger). Measured directly here instead
     (real-agent GUMBEL_UNTRAINED, 5 seeds, write_addr_tagger at its random
     PyTorch-default init, which is NOT the same init as query_proj/self.memory's
     custom small-scale init): per-seed Jaccard is highly seed-dependent
     (observed 0.0/0.5/1.0 at a 40-episode moderate-scale check) because an
     untrained tagger's argmin decision is a fixed but ARBITRARY function of
     its random weights -- some draws happen to separate the two well-
     separated synthetic clusters "for free" (low Jaccard, but NOT learned
     content-discrimination -- a generic property of applying any smooth,
     if random, function to two distant point clouds), others happen to lock
     (high Jaccard, mirroring the argmin single-slot pathology this whole
     substrate_queue entry is about). The MEAN across 5 seeds is what both
     GUMBEL_UNTRAINED_baseline_is_content_blind (descriptive) and C2 (gating)
     read, exactly as designed -- the mean is well-defined and reproducible
     regardless of what the individual-seed distribution looks like; only
     the (already-corrected) prose claiming a specific expected VALUE was
     wrong, not the measurement design itself.

Sanity/negative-control instrumentation (mechanical, real-agent-level
confirmation of the wiring, not just the unit contract's confirmation):
before/after state_dict comparison of write_addr_tagger's parameters across
the whole training run. GUMBEL_UNTRAINED is expected to show ZERO movement
(weight=0.0 -> the addressing loss term is never added -> zero gradient ever
reaches the tagger); GUMBEL_TRAINED is expected to show nonzero movement.
This is reported as a non-gating descriptive check, mirroring V3-EXQ-943's
own LEGACY reference-arm convention.

ACCEPTANCE CRITERIA -- BOTH required (registered here; substrate_queue's own
failure_record only carries the two-mechanism V3-EXQ-943 criterion, this
queue entry's `note` is the record for this one):
  C1_OCCUPANCY:   n_occupied_slots(GUMBEL_UNTRAINED) >= 2 on >= 3/5 seeds AND
                  n_occupied_slots(GUMBEL_TRAINED)   >= 2 on >= 3/5 seeds
                  (measured during real training; expected to pass trivially).
  C2_CONTENT_DISCRIMINATION: mean_jaccard(GUMBEL_TRAINED) <=
                  mean_jaccard(GUMBEL_UNTRAINED) - 0.25 (post-training
                  synthetic 2-cluster probe; MEAN across 5 seeds, not a
                  per-seed threshold -- test_refractory_preserves_content_
                  conditioning's own docstring documents why a per-seed
                  Jaccard margin at n=5 is unstable/sign-inconsistent and the
                  MEAN is the reproducible statistic; the 0.25 margin mirrors
                  that same contract's own margin convention for this
                  substrate_queue entry's family of measurements).
  PASS iff BOTH C1_OCCUPANCY and C2_CONTENT_DISCRIMINATION hold (given both
  readiness gates below). This is a genuinely open, falsifiable question:
  the architecture doc's own toy 300-step script did NOT move the loss off
  its near-uniform starting point, so a FAIL here (C1 passes, C2 does not) is
  a real, scientifically valid, expected-possible outcome, not a bug --
  exactly the same "SAME symmetry-breaking difficulty compute_diversification
  _loss() needed a full experiment to overcome" framing the doc uses. A
  40-episode moderate-scale check run during authoring (well short of the
  full 200-episode schedule, so not itself load-bearing) found the mechanical
  wiring correct -- GUMBEL_TRAINED's tagger moved substantially (max abs
  param diff ~0.68-0.76 across 5 seeds) while GUMBEL_UNTRAINED's stayed
  EXACTLY frozen (0.0 on 5/5) -- but the DIRECTIONAL result at that reduced
  scale was C2-FAIL (mean Jaccard moved from 0.4 untrained to 0.9 trained,
  i.e. WORSE/more collapsed, not better). Left unresolved deliberately: 40
  episodes is itself close to a "toy" scale by this file's own standard, and
  the whole reason this experiment exists is that toy-scale runs are not
  trustworthy for this specific question (see "WHY THIS RUN EXISTS" above).
  The full 200-episode run is what actually answers it.

P0 READINESS (gates C1's interpretability -- a below-floor writepath
engagement means "not ready", never a substrate verdict): every one of the 15
cells (3 arms x 5 seeds) must record n_write_calls >= WRITE_CALLS_FLOOR,
exactly V3-EXQ-943's own P0 gate (confirms sd016_writepath_mode="sense_only"
actually fired ContextMemory.write() during training on THIS run, rather than
assuming it from config alone).

C2 READINESS (gates C2's interpretability -- separate from P0 above, and
specific to this criterion): mean_jaccard(GUMBEL_UNTRAINED) must be >=
C2_JACCARD_MARGIN (0.25) for C2 to be a meaningful test at all -- otherwise
the untrained baseline is already close enough to the Jaccard=0 floor that
"materially lower" becomes structurally unsatisfiable regardless of what
training does, which would silently misread a floor effect as "training
failed." Per the corrected framing above, the untrained mean is NOT assumed
to be near 1.000, so this headroom is checked empirically rather than taken
for granted (measured 0.4 at the 40-episode moderate check, comfortably
above the 0.25 floor -- and reproducible at any training length, since
GUMBEL_UNTRAINED's tagger never receives gradient regardless of episode
count). If unmet at full scale, C2 self-routes to a distinct
`precondition_unmet` reading rather than a plain FAIL.

SUBSTRATE PROPERTIES held constant across all arms (not part of the
manipulation) -- identical to V3-EXQ-943's own list, reused verbatim so the
two experiments' LEGACY arms are the same measurement up to RNG:
  - contextmemory_gated_content_write=True (436d write-path repair).
  - sd016_writepath_mode="sense_only".
  - alpha_world=0.9 (SD-008).
  - use_noise_floor=True (MECH-313/ARC-065).
  - context_memory.memory.requires_grad_(False) after construction (436e/f
    Adam-drift neutralization). Note this does NOT freeze write_addr_tagger --
    that module's parameters are deliberately left trainable; only
    self.memory (the CONTENT tensor mutated by write()'s own .data write) is
    frozen against optimizer drift, exactly as in V3-EXQ-943.
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True.
  - use_salience_coordinator left at its REEConfig default (False). See
    "Step 2.5c" below.
  - sd016_enabled left at its REEConfig default (False) -- gates only the
    cue-tagger/context-divergence-loss apparatus, irrelevant here; verified
    against current e1_deep.py/agent.py exactly as V3-EXQ-943 verified it.

No sleep. Identical rationale to V3-EXQ-943: SD-016 sleep-cycle consolidation
is an orthogonal question from whether the write-ADDRESS mechanism itself
avoids the single-slot fixed point / achieves content-conditioning.
sleep_driver_pattern recorded as "N/A (no sleep loop)".

Step 2.5c substrate-path overlap (re-checked against the CURRENT
substrate_queue.json, 2026-08-28, not merely cited from V3-EXQ-943): open
entry `mode-governance-engagement` (severity=corrupting, status=
implemented_pending_validation) lists ree_core/agent.py among its
substrate_paths at whole-file granularity, which this driver necessarily
imports (REEAgent). Removed by construction, identical disposition to
V3-EXQ-943: this driver never sets use_salience_coordinator=True (REEConfig's
own False default is used throughout) and never reads agent.operating_mode or
any SalienceCoordinator output. `modulatory-bias-selection-authority` and
`SD-056` also name agent.py but both carry status "implemented" (not
"_pending_validation") in the current queue -- closed, non-blocking.
`contextmemory-write-path-addressing-degeneracy` itself (e1_deep.py) is the
entry under test here, not an unrelated one. No other open `corrupting`
entry's substrate_paths overlaps any module this driver imports or exercises.

Step 2.5b re-derive brake: N/A. claim_ids is empty (see below).

Step 2.4 GOV-REUSE-1 (existing-evidence check): the decisive readout (2-cluster
content-discrimination Jaccard for a TRAINED gumbel_learned write_addr_tagger
under real agent dynamics) is not recorded anywhere. The only prior
measurements are (a) the unit-level ContextMemory-direct contract
(test_contextmemory_write_gumbel_learned.py, no REEAgent, no environment,
untrained tagger only) and (b) V3-EXQ-943's real-agent validation, which
covers BIAS/REFRACTORY only and never constructs write_addr_tagger at all.
Neither is a compatible substrate/measurement for this question. Not
recoverable; proceeding to run.

claim_ids: [] -- this experiment tests substrate readiness for the
write-address FIX itself under real training dynamics (specifically:
does the redesigned pairwise-diversity addressing loss produce measurable
content-discrimination), not a claim hypothesis (SD-017 / ARC-045 / MECH-166,
the sleep-differentiation claims this substrate_queue entry unblocks once
validated). Diagnostic purpose experiments with claim_ids=[] are excluded
from governance confidence scoring by design.

Does NOT flip substrate_queue status regardless of outcome -- that stays a
human/governance disposition (see the substrate_queue entry's own
implementation_note convention). This script only appends a
validation_record once it has actually run.

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
from experiments._metrics import p0_readiness_gate, P0NotReady  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_956_contextmemory_write_gumbel_learned_validation"
QUEUE_ID = "V3-EXQ-956"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Arms --------------------------------------------------------------- #
# (label, write_selection, contextmemory_write_addressing_loss_weight)
ARMS: List[Tuple[str, str, float]] = [
    ("LEGACY", "argmin", 0.0),
    ("GUMBEL_UNTRAINED", "gumbel_learned", 0.0),
    ("GUMBEL_TRAINED", "gumbel_learned", 0.5),
]
ARM_NAMES = [a[0] for a in ARMS]

SEEDS: List[int] = [42, 7, 13, 100, 200]  # matches V3-EXQ-943 / 436-family

# Substrate properties held constant across all arms (not the manipulation).
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 100  # matches V3-EXQ-907's own real (non-toy) schedule
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000

NUM_SLOTS = 16  # ContextMemory default
LATENT_DIM = 64  # self_dim(32) + world_dim(32) -- must match the contract
                 # test's own LATENT_DIM for the post-training probe to be a
                 # like-for-like measurement.

# P0 readiness floor -- exactly V3-EXQ-943's own.
WRITE_CALLS_FLOOR = 200.0

# C1 occupancy criterion (mirrors V3-EXQ-943's registered acceptance floor).
C1_OCCUPIED_SLOTS_FLOOR = 2
C1_N_SEEDS_REQUIRED = 3

# C2 content-discrimination criterion (NEW for this mechanism -- see module
# docstring for why this is a mean-across-seeds margin, not per-seed).
C2_JACCARD_MARGIN = 0.25

# Post-training synthetic 2-cluster probe (reuses the contract test's own
# instrument -- test_contextmemory_write_address_selection.py's _stream).
PROBE_N = 1500
PROBE_JITTER = 0.0078
PROBE_CLUSTERS = 2

TAGGER_MOVE_EPS = 1e-8  # threshold for "did write_addr_tagger's state_dict move"


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused verbatim from V3-EXQ-943)         #
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


def _make_agent(env: CausalGridWorldV2, write_selection: str,
                 write_addressing_loss_weight: float) -> REEAgent:
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
        contextmemory_write_addressing_loss_weight=write_addressing_loss_weight,
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
        "update NUM_SLOTS before trusting occupancy-floor thresholds."
    )
    total_latent_dim = agent.e1.config.self_dim + agent.e1.config.world_dim
    assert total_latent_dim == LATENT_DIM, (
        f"self_dim+world_dim={total_latent_dim} != LATENT_DIM={LATENT_DIM} -- "
        "the post-training synthetic probe would no longer match the "
        "contract test's own instrument; update LATENT_DIM before trusting "
        "the content-discrimination measurement."
    )
    if write_selection == "gumbel_learned":
        assert agent.e1.context_memory.write_addr_tagger is not None, (
            "write_selection='gumbel_learned' did not construct "
            "write_addr_tagger -- config wiring regression."
        )
    else:
        assert agent.e1.context_memory.write_addr_tagger is None, (
            f"write_selection={write_selection!r} constructed write_addr_tagger "
            "-- it should only exist in gumbel_learned mode."
        )
    # 436e/f Adam-drift neutralization (see module docstring). write() still
    # mutates memory.data directly under its own no_grad block regardless.
    # write_addr_tagger is deliberately NOT frozen -- it is this mechanism's
    # own trainable parameter, the entire subject of the manipulation.
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
    """Temperature-graded softmax sample over predicted harm (low harm -> high
    selection probability), using the MECH-313 noise-floor effective
    temperature. Reused verbatim from the 436/943 lineage's baseline policy.
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
# Write-address instrumentation -- INSTRUMENTATION FIELDS ONLY, no          #
# re-derivation of the selection expression (architecture doc mandate).    #
# ------------------------------------------------------------------ #

class _WriteSequenceTracker:
    """Records the ordered sequence of slots ContextMemory.write() actually
    mutated, by polling ContextMemory.last_write_index / .slot_write_counts
    (the authoritative instrumentation, maintained in every selection mode
    including legacy and gumbel_learned) after every agent.sense() call --
    never by recomputing write()'s own scoring expression. Identical to
    V3-EXQ-943's tracker.
    """

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


def _compute_deterministic_dvs(sequence: List[int], num_slots: int) -> Dict[str, Any]:
    """The deterministic occupancy-family columns V3-EXQ-943 established as
    robust -- NOT occ_cos, NOT the synthetic-probe Jaccard (that is a SEPARATE
    post-training measurement, see _run_content_probe below).
    """
    n_write_calls = len(sequence)
    if n_write_calls == 0:
        return {
            "n_write_calls": 0,
            "n_occupied_slots": 0,
            "entropy_bits": 0.0,
            "self_repeat_rate": None,
            "round_robin_agreement": None,
            "slot_write_counts": [0] * num_slots,
        }

    counts = [0] * num_slots
    for s in sequence:
        counts[s] += 1
    n_occupied = sum(1 for c in counts if c > 0)

    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / n_write_calls
            entropy -= p * math.log2(p)

    if n_write_calls >= 2:
        repeats = sum(
            1 for i in range(1, n_write_calls) if sequence[i] == sequence[i - 1]
        )
        self_repeat_rate = repeats / (n_write_calls - 1)
    else:
        self_repeat_rate = None

    last_write_tick = [-1] * num_slots
    agree = 0
    for tick, s in enumerate(sequence):
        lru = min(range(num_slots), key=lambda i: last_write_tick[i])
        if s == lru:
            agree += 1
        last_write_tick[s] = tick
    round_robin_agreement = agree / n_write_calls

    return {
        "n_write_calls": n_write_calls,
        "n_occupied_slots": n_occupied,
        "entropy_bits": entropy,
        "self_repeat_rate": self_repeat_rate,
        "round_robin_agreement": round_robin_agreement,
        "slot_write_counts": counts,
    }


# ------------------------------------------------------------------ #
# Post-training content-discrimination probe. Reuses the CONTRACT TEST'S   #
# OWN 2-cluster synthetic instrument (test_contextmemory_write_address_    #
# selection.py's _stream/_jaccard) so the measurement is directly          #
# comparable to the unit-level ~1.000 content-blind baseline.              #
# ------------------------------------------------------------------ #

def _probe_stream(seed: int, n: int, latent_dim: int, jitter: float, clusters: int):
    gen = torch.Generator().manual_seed(seed)
    bases = [torch.randn(1, latent_dim, generator=gen) * 0.078 for _ in range(clusters)]
    return [
        (i % clusters, bases[i % clusters] + torch.randn(1, latent_dim, generator=gen) * jitter)
        for i in range(n)
    ]


def _jaccard(per_cluster: Dict[int, Set[int]]) -> float:
    a, b = per_cluster.get(0, set()), per_cluster.get(1, set())
    return len(a & b) / max(len(a | b), 1)


def _run_content_probe(agent: REEAgent, seed: int) -> Tuple[float, Dict[int, Set[int]]]:
    """Post-training, EVAL-MODE probe. eval() makes gumbel_learned selection
    deterministic argmin on write_addr_tagger's own (possibly-trained) scores
    -- exactly the isolation the architecture doc specifies, so noise never
    contaminates what this measures. LEGACY is deterministic in every mode
    regardless, so eval() is a no-op for it. Uses a probe-specific seed
    offset so this measurement's RNG stream never collides with the training
    seed's own consumption earlier in the same cell.
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


# ------------------------------------------------------------------ #
# Episode / cell runners                                                   #
# ------------------------------------------------------------------ #

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    optimizer,
    harm_eval_opt,
    harm_buf_pos: List[torch.Tensor],
    harm_buf_neg: List[torch.Tensor],
    tracker: _WriteSequenceTracker,
) -> None:
    """Run a single training episode: sense -> E1 tick -> baseline action
    selection -> step -> prediction-loss training (which now includes the
    write-addressing loss when weight>0 and mode==gumbel_learned, wired
    inside agent.compute_prediction_loss() -- see ree_core/agent.py) ->
    harm_eval training.

    DELIBERATE DIVERGENCE FROM V3-EXQ-943's episode runner: this driver
    additionally calls `agent._e1_tick(latent)` after `agent.clock.advance()`
    -- V3-EXQ-943's own runner discarded the ticks dict and never called it,
    which was harmless THERE because BIAS/REFRACTORY occupancy comes
    entirely from sense()'s own SD-016 per-tick write hook, independent of
    _e1_tick. It is NOT harmless here: `compute_prediction_loss()` (below)
    returns an unconditional `zero_loss` whenever `self._world_experience_
    buffer` has fewer than 2 entries (ree_core/agent.py:10223-10225), and
    that buffer is populated ONLY inside `_e1_tick` (ree_core/agent.py:5469)
    -- which itself only runs from `act()`/`act_with_split_obs()`, neither of
    which this driver (or 943's) ever calls, since both use a hand-rolled
    baseline action-selection policy instead of the agent's own select_action.
    Confirmed empirically before landing this fix: with _e1_tick unwired,
    `agent.compute_prediction_loss()` returned exactly 0.0 on every single
    step of a 20-step probe, so the addressing loss (gated on `weight>0 AND
    write_selection=="gumbel_learned"`, added to `loss` INSIDE
    compute_prediction_loss) was silently never added to the backward graph,
    and GUMBEL_TRAINED's write_addr_tagger showed byte-identical zero
    movement from init across a real 40-episode moderate-scale check --
    exactly as if weight were 0.0. The call pattern below (advance -> capture
    ticks -> conditionally call _e1_tick) is the exact convention
    experiments/_harness.py uses for every OTHER diagnostic that needs a real
    E1 prediction-loss gradient under a custom action-selection policy; the
    e1_prior return value itself is discarded here (unlike _harness.py) --
    this driver never calls generate_trajectories/select_action, so nothing
    downstream reads it.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        ticks = agent.clock.advance()
        if ticks.get("e1_tick", False):
            agent._e1_tick(latent)
        tracker.poll(agent.e1.context_memory)  # exact -- see class docstring

        z_world = latent.z_world.detach().clone()
        action_idx = _select_action_baseline(agent, z_world, env.action_dim)
        action_oh = _action_onehot(action_idx, env.action_dim, agent.device)
        agent._last_action = action_oh

        _, harm_signal, done, _info, obs_dict = env.step(action_oh)
        is_harm = float(harm_signal) < 0

        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        total = e1_loss + e2_loss
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


def _run_cell(arm_name: str, write_selection: str, write_addressing_loss_weight: float,
              seed: int, base_config_slice: Dict[str, Any],
              zg: ZGoalStreamAccumulator, training_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "arm": arm_name,
        "contextmemory_write_selection": write_selection,
        "contextmemory_write_addressing_loss_weight": write_addressing_loss_weight,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe, write_selection, write_addressing_loss_weight)

        tagger_init_state = None
        if write_selection == "gumbel_learned":
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
        tracker = _WriteSequenceTracker()

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            _run_episode(
                agent, env, steps_per_episode, optimizer, harm_eval_opt,
                harm_buf_pos, harm_buf_neg, tracker,
            )
            if (ep + 1) % 50 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] arm={arm_name} seed={seed} "
                    f"ep {ep + 1}/{training_episodes} "
                    f"n_writes={len(tracker.sequence)}",
                    flush=True,
                )

        zg.observe(agent)
        dvs = _compute_deterministic_dvs(tracker.sequence, NUM_SLOTS)

        tagger_params_moved: Optional[bool] = None
        if write_selection == "gumbel_learned" and tagger_init_state is not None:
            final_state = agent.e1.context_memory.write_addr_tagger.state_dict()
            max_abs_diff = 0.0
            for k, v0 in tagger_init_state.items():
                d = (final_state[k] - v0).abs().max().item()
                max_abs_diff = max(max_abs_diff, d)
            tagger_params_moved = max_abs_diff > TAGGER_MOVE_EPS
        else:
            max_abs_diff = None

        probe_jaccard, probe_per_cluster = _run_content_probe(agent, seed)

        row: Dict[str, Any] = {
            "arm": arm_name,
            "seed": seed,
            "write_selection": write_selection,
            "contextmemory_write_addressing_loss_weight": write_addressing_loss_weight,
            **dvs,
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_2cluster_jaccard": probe_jaccard,
            "probe_2cluster_occupied": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster.items()
            },
        }
        cell.stamp(row)

    passed = dvs["n_occupied_slots"] >= C1_OCCUPIED_SLOTS_FLOOR
    print(
        f"verdict: {'PASS' if passed else 'FAIL'} "
        f"(occupancy; probe_jaccard={probe_jaccard:.3f})",
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
    seeds = SEEDS[:1] if dry_run else SEEDS

    full_config_slice: Dict[str, Any] = {
        "arms": ARM_NAMES,
        "seeds": seeds,
        "training_episodes": training_episodes,
        "steps_per_episode": steps_per_episode,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
    }

    arm_results: List[Dict[str, Any]] = []
    for arm_name, wsel, waw in ARMS:
        for seed in seeds:
            row = _run_cell(
                arm_name, wsel, waw, seed, full_config_slice, zg,
                training_episodes, steps_per_episode,
            )
            arm_results.append(row)

    by_arm: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ARM_NAMES}
    for row in arm_results:
        by_arm[row["arm"]].append(row)

    # --- P0 readiness: writepath genuinely engaged in every cell ---
    write_call_counts = [row["n_write_calls"] for row in arm_results]
    min_write_calls = min(write_call_counts) if write_call_counts else 0.0
    try:
        preconditions = p0_readiness_gate([
            {
                "name": "writepath_engaged_every_cell",
                "measured": float(min_write_calls),
                "threshold": WRITE_CALLS_FLOOR,
                "direction": "lower",
            },
        ])
    except P0NotReady as e:
        status = "FAIL"
        label = "substrate_not_ready_requeue"
        result: Dict[str, Any] = {
            "outcome": status,
            "status": status,
            "claim_ids": CLAIM_IDS,
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "evidence_direction": "non_contributory",
            "sleep_driver_pattern": "N/A (no sleep loop)",
            "arm_results": arm_results,
            "interpretation": {
                "label": label,
                "preconditions": e.preconditions,
                "criteria_non_degenerate": {},
            },
            "fatal_error_count": 0,
        }
        return result, zg

    # --- C1: occupancy floor, applied to GUMBEL_UNTRAINED and GUMBEL_TRAINED ---
    def _seeds_clearing_occupancy_floor(arm_name: str) -> int:
        return sum(
            1 for row in by_arm[arm_name]
            if row["n_occupied_slots"] >= C1_OCCUPIED_SLOTS_FLOOR
        )

    n_pass_untrained = _seeds_clearing_occupancy_floor("GUMBEL_UNTRAINED")
    n_pass_trained = _seeds_clearing_occupancy_floor("GUMBEL_TRAINED")
    n_pass_legacy = _seeds_clearing_occupancy_floor("LEGACY")  # descriptive only

    c1_untrained_pass = n_pass_untrained >= C1_N_SEEDS_REQUIRED
    c1_trained_pass = n_pass_trained >= C1_N_SEEDS_REQUIRED
    c1_pass = c1_untrained_pass and c1_trained_pass

    # --- C2: content-discrimination, mean Jaccard across seeds ---
    def _mean_probe_jaccard(arm_name: str) -> float:
        vals = [row["probe_2cluster_jaccard"] for row in by_arm[arm_name]]
        return sum(vals) / len(vals) if vals else float("nan")

    mean_jaccard_legacy = _mean_probe_jaccard("LEGACY")
    mean_jaccard_untrained = _mean_probe_jaccard("GUMBEL_UNTRAINED")
    mean_jaccard_trained = _mean_probe_jaccard("GUMBEL_TRAINED")

    # C2 readiness (module docstring "C2 READINESS"): the untrained baseline
    # must have enough room below it for "materially lower" to be satisfiable
    # at all -- otherwise a FAIL would be a floor effect, not evidence about
    # training. Checked empirically rather than assumed (see the "CORRECTION
    # TO AN EARLIER ASSUMPTION" note above).
    c2_headroom_met = mean_jaccard_untrained >= C2_JACCARD_MARGIN
    c2_directional_pass = mean_jaccard_trained <= (mean_jaccard_untrained - C2_JACCARD_MARGIN)
    c2_pass = c2_headroom_met and c2_directional_pass

    overall_pass = c1_pass and c2_pass

    # --- Descriptive, non-gating negative controls ---
    n_untrained_seeds_frozen = sum(
        1 for row in by_arm["GUMBEL_UNTRAINED"] if row["tagger_params_moved"] is False
    )
    n_trained_seeds_moved = sum(
        1 for row in by_arm["GUMBEL_TRAINED"] if row["tagger_params_moved"] is True
    )

    criteria = [
        {
            "name": "C1_GUMBEL_UNTRAINED_occupancy_floor",
            "load_bearing": True,
            "passed": c1_untrained_pass,
            "n_seeds_passed": n_pass_untrained,
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
        },
        {
            "name": "C1_GUMBEL_TRAINED_occupancy_floor",
            "load_bearing": True,
            "passed": c1_trained_pass,
            "n_seeds_passed": n_pass_trained,
            "n_seeds_required": C1_N_SEEDS_REQUIRED,
        },
        {
            "name": "C2_readiness_untrained_baseline_headroom",
            "load_bearing": False,
            "kind": "readiness",
            "passed": c2_headroom_met,
            "measured": mean_jaccard_untrained,
            "threshold": C2_JACCARD_MARGIN,
            "direction": "lower",
            "note": (
                "Precondition for C2's interpretability, not part of the "
                "manipulation itself: mean_jaccard(GUMBEL_UNTRAINED) must be "
                ">= 0.25 for 'materially lower' to be satisfiable at all. If "
                "unmet, C2 below reads precondition_unmet rather than a "
                "plain FAIL -- a floor effect is not evidence training failed."
            ),
        },
        {
            "name": "C2_content_discrimination_improves_with_training",
            "load_bearing": True,
            "passed": c2_pass,
            "precondition_unmet": not c2_headroom_met,
            "directional_result": c2_directional_pass,
            "mean_jaccard_gumbel_untrained": mean_jaccard_untrained,
            "mean_jaccard_gumbel_trained": mean_jaccard_trained,
            "required_margin": C2_JACCARD_MARGIN,
            "note": (
                "PASS iff the C2 readiness precondition is met AND "
                "mean_jaccard(GUMBEL_TRAINED) <= mean_jaccard(GUMBEL_UNTRAINED) "
                "- 0.25. This is the genuinely open question; a FAIL here "
                "(precondition met, directional_result False) is a real, "
                "expected-possible scientific outcome (see module "
                "docstring), not a bug. A FAIL with precondition_unmet=True "
                "means the comparison could not be made at all, not that "
                "training was tried and failed."
            ),
        },
        {
            "name": "LEGACY_reference_reproduces_known_degeneracy",
            "load_bearing": False,
            "passed": n_pass_legacy < C1_N_SEEDS_REQUIRED,
            "n_seeds_clearing_floor": n_pass_legacy,
            "note": "Descriptive only, matches V3-EXQ-943's own LEGACY reference convention.",
        },
        {
            "name": "GUMBEL_UNTRAINED_mean_jaccard_observed",
            "load_bearing": False,
            "passed": True,
            "mean_jaccard": mean_jaccard_untrained,
            "note": (
                "Purely descriptive report of the untrained baseline's mean "
                "Jaccard -- always 'passed' (this is a measurement, not a "
                "criterion). See the module docstring's 'CORRECTION TO AN "
                "EARLIER ASSUMPTION' for why no specific value (e.g. ~1.000) "
                "is expected a priori; C2_readiness_untrained_baseline_"
                "headroom above is the actual gate on this quantity."
            ),
        },
        {
            "name": "GUMBEL_UNTRAINED_tagger_received_zero_gradient",
            "load_bearing": False,
            "passed": n_untrained_seeds_frozen == len(by_arm["GUMBEL_UNTRAINED"]),
            "n_seeds_frozen": n_untrained_seeds_frozen,
            "n_seeds_total": len(by_arm["GUMBEL_UNTRAINED"]),
            "note": (
                "Mechanical negative control: weight=0.0 should leave "
                "write_addr_tagger's state_dict byte-identical to its random "
                "init across the whole real training run (the addressing "
                "loss term is gated on weight>0 in agent.py, so it is never "
                "added to the backward graph). Confirms the wiring at the "
                "REAL-AGENT level, not just the unit contract."
            ),
        },
        {
            "name": "GUMBEL_TRAINED_tagger_received_gradient",
            "load_bearing": False,
            "passed": n_trained_seeds_moved == len(by_arm["GUMBEL_TRAINED"]),
            "n_seeds_moved": n_trained_seeds_moved,
            "n_seeds_total": len(by_arm["GUMBEL_TRAINED"]),
            "note": (
                "Mechanical positive control: weight=0.5 should move "
                "write_addr_tagger's parameters away from init over the "
                "real training run, regardless of whether that movement "
                "achieves content-discrimination (C2)."
            ),
        },
    ]
    criteria_non_degenerate = {
        "C1_GUMBEL_UNTRAINED_occupancy_floor": len(seeds) >= 2,
        "C1_GUMBEL_TRAINED_occupancy_floor": len(seeds) >= 2,
        "C2_content_discrimination_improves_with_training": (
            len(seeds) >= 2 and c2_headroom_met
        ),
    }

    if overall_pass:
        label = "gumbel_learned_content_discrimination_validated_under_real_training"
    elif c1_pass and not c2_headroom_met:
        label = "gumbel_learned_c2_precondition_unmet_untrained_baseline_floor_effect"
    elif c1_pass and not c2_pass:
        label = "gumbel_learned_occupancy_only_content_discrimination_not_confirmed"
    else:
        label = "gumbel_learned_write_address_fix_not_confirmed_under_real_agent"
    # claim_ids is empty -- this diagnostic validates a SUBSTRATE mechanism,
    # not a claim hypothesis, so it cannot "support" or "weaken" a claim
    # regardless of PASS/FAIL (matches V3-EXQ-943's own convention).
    evidence_direction = "non_contributory"

    status = "PASS" if overall_pass else "FAIL"

    metrics: Dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        rows = by_arm[arm_name]
        metrics[f"n_occupied_slots_per_seed_{arm_name}"] = [r["n_occupied_slots"] for r in rows]
        metrics[f"n_write_calls_per_seed_{arm_name}"] = [r["n_write_calls"] for r in rows]
        metrics[f"entropy_bits_per_seed_{arm_name}"] = [r["entropy_bits"] for r in rows]
        metrics[f"self_repeat_rate_per_seed_{arm_name}"] = [r["self_repeat_rate"] for r in rows]
        metrics[f"round_robin_agreement_per_seed_{arm_name}"] = [r["round_robin_agreement"] for r in rows]
        metrics[f"probe_2cluster_jaccard_per_seed_{arm_name}"] = [r["probe_2cluster_jaccard"] for r in rows]
        metrics[f"n_seeds_clearing_occupancy_floor_{arm_name}"] = _seeds_clearing_occupancy_floor(arm_name)
        metrics[f"mean_probe_2cluster_jaccard_{arm_name}"] = _mean_probe_jaccard(arm_name)

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory gumbel_learned write-address selection, "
        f"real-agent validation\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; validates substrate_queue "
        f"contextmemory-write-path-addressing-degeneracy, THIRD mechanism)\n\n"
        f"| Arm | seeds >= {C1_OCCUPIED_SLOTS_FLOOR} occupied | mean probe Jaccard | "
        f"n_write_calls range |\n"
        f"|---|---|---|---|\n"
        + "".join(
            f"| {arm_name} | {_seeds_clearing_occupancy_floor(arm_name)}/{len(by_arm[arm_name])} "
            f"| {_mean_probe_jaccard(arm_name):.3f} "
            f"| {min(r['n_write_calls'] for r in by_arm[arm_name])}-"
            f"{max(r['n_write_calls'] for r in by_arm[arm_name])} |\n"
            for arm_name in ARM_NAMES
        )
        + f"\nC1 (occupancy, both GUMBEL arms): {'PASS' if c1_pass else 'FAIL'}\n"
        f"C2 (content-discrimination, mean Jaccard trained <= untrained - "
        f"{C2_JACCARD_MARGIN}): {'PASS' if c2_pass else 'FAIL'} "
        f"(untrained={mean_jaccard_untrained:.3f}, trained={mean_jaccard_trained:.3f})\n"
    )

    result = {
        "outcome": status,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": evidence_direction,
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
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
        "arms": ARM_NAMES,
        "seeds": SEEDS,
        "training_episodes": TRAINING_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "c1_occupied_slots_floor": C1_OCCUPIED_SLOTS_FLOOR,
        "c1_n_seeds_required": C1_N_SEEDS_REQUIRED,
        "c2_jaccard_margin": C2_JACCARD_MARGIN,
        "write_calls_floor": WRITE_CALLS_FLOOR,
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
