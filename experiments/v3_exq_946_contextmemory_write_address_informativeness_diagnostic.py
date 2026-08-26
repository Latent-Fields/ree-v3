#!/opt/local/bin/python3
"""
V3-EXQ-946 -- ContextMemory write-ADDRESS informativeness: is there any
operating point (of the two landed, default-off occupancy fixes) at which
the write address is CONTEXT-determined, rather than merely order/period-
determined, on a real REEAgent loop?

Chip: chip-20260823-queueexp-contextmemory-write-addressing-diagnostic

experiment_purpose: diagnostic

WHY THIS RUN EXISTS. V3-EXQ-943 (PASS, 2026-08-20) validated write
OCCUPANCY under a real agent for both landed fixes (conscience-bias
`contextmemory_write_usage_balancing` and eligibility-mask `refractory`
write_selection) -- but did NOT test whether the resulting address carries
any information about CONTEXT. Governance recorded this gap itself
(substrate_queue.json entry `contextmemory-write-path-addressing-
degeneracy`, field `validation_record_943`, 2026-08-21T17:05:43Z, session
gov-20260821-0203): BIAS's round_robin_agreement of 0.993-0.996 is a
least-recently-used CLOCK ("occupancy-without-addressing"), and
REFRACTORY's entropy of exactly log2(3) on the locking seeds is the k+1
construction floor -- neither result speaks to whether the SLOT SELECTED
carries any information about the CONTEXT the agent was in when it wrote.
This is the open blocker for EVB-0628 (INV-044, marked blocked_substrate
2026-08-23, REE_assembly 54e114c5de): no attribution/differentiation DV
over slot assignment can carry signal until write-address informativeness
is established at some operating point.

THE METHODOLOGICAL TRAP THIS EXPERIMENT IS BUILT TO AVOID. Scoring
I(slot;context) against a CHANCE baseline is not sufficient. A
least-recently-used clock (period num_slots=16) or a period-(k+1)
refractory cycle can produce non-zero mutual information purely from
period ALIGNMENT with the context block schedule (CONTEXT_SWITCH_EVERY),
with zero actual context-sensitivity in the addressing mechanism itself --
this is the artifact-as-verdict error that has cost the
contextmemory-write-path-addressing-degeneracy lineage six generations
(V3-EXQ-436..436f) and got both of INV-044's experimental entries
withdrawn on 2026-08-22. So the load-bearing gate here is NOT "does
I(slot;context) exceed chance" -- it is "does I(slot;context) exceed an
ORDER-ONLY NULL": the same observed write sequence, with context labels
permuted BLOCKWISE (preserving block structure/autocorrelation -- which
block boundaries a write fell in -- and only breaking the assignment of
WHICH blocks were safe vs dangerous). A purely order-driven address (a
clock, a period-(k+1) cycle) produces the SAME mutual information under
that permutation as under the true labels, because its address sequence
never actually reads context; only a genuinely context-sensitive address
would show excess MI over that null.

ARMS. BIAS is swept over contextmemory_write_usage_bias_weight (already
wired at all three REEConfig sites -- E1Config field, from_dims parameter,
from_dims assignment -- config-only, no substrate build) at {1.0 (the
V3-EXQ-943 default), 0.1, 0.01}, to ask whether WEAKENING the usage term
relative to the raw content-similarity term (mean_scores) lets more of the
address's content-dependence survive, since at the default weight the
usage term dominates content by 2-3 orders of magnitude (E1Config's own
sqrt(memory_dim) scaling comment) and content is the only channel through
which context could plausibly reach the address (there is no direct
context input to write() at all -- see "content is the only channel" below).

REFRACTORY IS NOT CROSSED WITH bias_weight -- DEVIATION FROM THE LITERAL
BRIEF, NECESSITY-JUSTIFIED (Step 2.4/3.5 "necessity" + DV-symmetry
declaration, see module source: `ContextMemory.write()`,
`ree_core/predictors/e1_deep.py`). `write_usage_bias_weight` is read in
exactly one place, inside `if self.write_usage_balancing:` -- REFRACTORY
sets `contextmemory_write_usage_balancing=False`, so the parameter is
stored on the module but NEVER READ by REFRACTORY's selection path
(`_select_write_slot` for `write_selection="refractory"` masks the last k
slots and argmins the UNMODIFIED `mean_scores`; it never touches `bias`).
Three REFRACTORY cells at bias_weight in {1.0, 0.1, 0.01} and the same
seed would therefore be bit-identical -- the manipulation is invariant
under the arm's own mechanism by construction, precisely the DV-symmetry
degeneracy this skill's design-audit step exists to catch (parallel case:
V3-EXQ-604c's broadcast-scalar-under-argmax). REFRACTORY instead runs
once per seed at k=2 (V3-EXQ-943's validated operating point, matching
occupancy-floor precedent) -- 5 cells, not 15. Total: 4 arms (3 BIAS
sub-arms + 1 REFRACTORY) x 5 seeds = 20 cells, vs. the 15 a literal 2x3
cross would have run at 3 of them being redundant compute for zero new
information.

CONTENT IS THE ONLY CHANNEL THROUGH WHICH CONTEXT COULD REACH THE
ADDRESS. `write()`'s query (`query_proj(state)`) and payload
(`write_gate(state) * write_content(state)`) are both functions of the
per-tick agent `state` (which folds in z_world, and hence anything
upstream that differs between the safe and dangerous environments), so
context enters the address computation only insofar as `state` differs by
context AND that difference survives into `mean_scores`'s cross-slot
spread. Lowering `bias_weight` cannot CREATE a context signal that is not
already present in `mean_scores` -- it can only stop MASKING one that is.
So a null result at all three weights is informative in a specific way:
it says the content channel itself carries no discriminable context
signal at this operating point, not merely that the usage term drowned a
signal that would otherwise show up.

MEASUREMENT INSTRUMENT VALIDATION (P0, before any real cell runs -- see
"P0 READINESS" below). Because the load-bearing criterion here is itself
a NEW measured quantity (mutual information against a purpose-built null)
rather than a previously-validated substrate check, the skill's P0
readiness-assert rule is satisfied by validating the INSTRUMENT (the
MI-vs-order-only-null test itself) on two synthetic controls built from a
canonical tick/block schedule with the same structure a real cell has
(no agent, no environment -- pure arithmetic, so this costs milliseconds):
  - POSITIVE CONTROL: a synthetic address sequence constructed to be a
    pure function of the TRUE context label (slot 0 if context==0, slot 1
    otherwise) -- zero content of order per se. Its MI must clear a
    (deliberately large) z-score floor against its own order-only null,
    proving the pipeline CAN detect a real context-address relationship
    when one exists.
  - NEGATIVE CONTROL: a synthetic address sequence constructed to be a
    pure function of TICK INDEX alone (round-robin, period=NUM_SLOTS) --
    zero context content, by construction indistinguishable from
    REFRACTORY/BIAS's own order-driven signature. Its z-score against its
    own order-only null must stay UNDER a ceiling, proving the null
    construction correctly nets out period-alignment artifacts rather
    than flagging a pure clock as context-informative -- this is the
    exact failure mode (six-generation artifact-as-verdict lineage,
    above) the null exists to prevent, tested directly rather than
    assumed.
Both are readiness-kind preconditions (measured z-score, explicit
threshold, explicit control) per the skill's P0 readiness-assert rule.
Unlike a substrate-not-ready failure, an unmet instrument-validation
precondition here means the MEASUREMENT ITSELF is not trustworthy, so it
self-routes to `mi_instrument_not_validated` (never `substrate_not_ready_
requeue`, which would misattribute the failure to ContextMemory readiness
rather than to this driver's own statistic).

P0 READINESS (three preconditions, ALL gate the run before any of the 20
real cells is interpreted):
  1. writepath_engaged_every_cell -- min pooled n_write_calls across all
     20 cells >= WRITE_CALLS_FLOOR (200; V3-EXQ-943's own floor, that run
     saw 2900-3200 per cell under the identical harness).
  2. mi_null_test_detects_positive_control -- z-score of the synthetic
     context-pure address vs its own order-only null >= 5.0.
  3. mi_null_test_rejects_negative_control -- z-score of the synthetic
     order-pure address vs its own order-only null <= 2.0 (ceiling).
Any unmet -> FAIL, label per which precondition failed (see above), no
per-cell criteria computed.

DELIVERABLE / LOAD-BEARING CRITERION. Per arm (BIAS_W1_0, BIAS_W0_1,
BIAS_W0_01, REFRACTORY), per seed: compute I(slot;context) exactly from
the (slot, context) contingency table over the recorded write sequence,
and an order-only null distribution (500 blockwise permutations of which
blocks are safe vs dangerous, block STRUCTURE held fixed) for that same
sequence. A seed CLEARS the bar if z_vs_null = (observed - null_mean) /
null_std >= Z_NULL_EXCEED_THRESHOLD (2.0, pre-registered, ~one-tailed
p<0.023). An arm PASSES if >=3/5 seeds clear the bar (mirrors V3-EXQ-943's
own >=3/5-seeds convention for continuity, NOT inherited from
substrate_queue's registered occupancy floor -- that floor is about
occupancy, this bar is newly pre-registered here for addressing). Overall
`outcome` is PASS iff AT LEAST ONE arm clears its bar (i.e. this run
found SOME operating point with a context-informative address); FAIL
means no operating point tested shows an address exceeding the
order-only null anywhere -- itself the informative, falsifiable answer
the substrate_queue entry and EVB-0628 need, not an experiment failure.

DETERMINISTIC REFERENCE COLUMNS (unchanged from V3-EXQ-943): kept for
comparability across the two runs, and per the pre-registered probe
(contextmemory_write_selection_comparison_20260819.md) NOT the load-
bearing DV here (that probe found occ_cos and cluster Jaccard cannot
discriminate BIAS/REFRACTORY at 5 seeds -- neither is computed by this
script either): n_occupied_slots, self_repeat_rate, entropy_bits,
round_robin_agreement.

Step 2.4 GOV-REUSE-1 (existing-evidence check): the decisive readout
(I(slot;context) against an order-only null, under a real agent) is NOT
recorded anywhere. V3-EXQ-943's own manifest records only aggregate
per-seed columns (n_occupied_slots, entropy_bits, self_repeat_rate,
round_robin_agreement, slot_write_counts) -- confirmed by inspecting its
arm_results[0] keys directly -- never the raw per-write (slot, context)
sequence this diagnostic's DV needs, so 943's manifest cannot be
reanalyzed post-hoc for this question. Not recoverable; proceeding to run.

Step 2.5b re-derive brake: N/A. claim_ids is empty -- this experiment
tests substrate/instrument readiness, not a claim hypothesis, so no
claim's autopsy count applies (identical disposition to V3-EXQ-943).

Step 2.5c substrate-path overlap (checked against the 2026-08-23
substrate_queue.json): two open `corrupting` entries name a module this
driver imports.
  - `contextmemory-write-path-addressing-degeneracy` (e1_deep.py) IS the
    entry this diagnostic exists to inform, not an unrelated defect --
    same disposition as V3-EXQ-943.
  - `mode-governance-engagement` (agent.py, among others, whole-file
    granularity) -- removed by construction, identical disposition to
    V3-EXQ-943 and V3-EXQ-942 before it: this driver never sets
    use_salience_coordinator=True (REEConfig's own default, False, used
    throughout) and never reads agent.operating_mode or any
    SalienceCoordinator output.
No other open corrupting entry's substrate_paths overlaps any module this
driver imports or exercises.

No sleep. use_sleep_loop / sws_enabled / rem_enabled all left at their
REEConfig defaults (False) -- sleep-cycle consolidation is an orthogonal
question (the SD-017/ARC-045/MECH-166 sleep-differentiation claims) from
whether the WRITE-ADDRESS mechanism carries context information.
sleep_driver_pattern recorded as "N/A (no sleep loop)".

claim_ids: [] -- this experiment establishes measurement readiness for
EVB-0628 (INV-044); it is NOT evidence for or against any claim.
Diagnostic-purpose experiments with claim_ids=[] are excluded from
governance confidence scoring by design.

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
import hashlib
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


EXPERIMENT_TYPE = "v3_exq_946_contextmemory_write_address_informativeness_diagnostic"
QUEUE_ID = "V3-EXQ-946"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# validate_experiments.py readiness-anchor reachability check (advisory): both
# mi_null_test_detects_positive_control / mi_null_test_rejects_negative_control
# are reachable BY CONSTRUCTION, not by replicating a frozen historical
# reference cell -- they score a purely SYNTHETIC, deterministically-generated
# sequence (no agent, no environment, no substrate dependence at all) through
# the exact same score_fn (_z_vs_null + _mutual_information_bits) the real
# cells use. Verified numerically before shipping (not asserted blindly):
# positive control z=40.19 (>> the 5.0 floor; context perfectly separates 2 of
# 16 slots, so observed MI=1.0 bit against a near-zero blockwise-permutation
# null), negative control z=-0.27 (<< the 2.0 ceiling; a pure period-16 clock
# against 40 blocks of ~75 ticks each shows no excess over its own null). Both
# margins are wide (8x and 7x respectively), so neither threshold is fragile.
# No prior run exists to supersede (this script has not run before).
ANCHOR_REACHABILITY_EXEMPT = (
    "both anchors score a synthetic, deterministically-constructed sequence "
    "(no agent/environment/substrate dependence) through the shipped score_fn; "
    "reachable by construction, verified numerically at authoring time: "
    "positive z=40.19 vs floor 5.0, negative z=-0.27 vs ceiling 2.0 -- see "
    "comment above for the derivation"
)

# --- Arms --------------------------------------------------------------- #
# (label, write_usage_balancing, write_selection, write_refractory_k, bias_weight)
# bias_weight is meaningless for REFRACTORY (write_usage_balancing=False means
# it is never read -- see module docstring "REFRACTORY IS NOT CROSSED"); the
# value 1.0 is recorded there purely for schema uniformity, never exercised.
ARMS: List[Tuple[str, bool, str, int, float]] = [
    ("BIAS_W1_0", True, "argmin", 2, 1.0),
    ("BIAS_W0_1", True, "argmin", 2, 0.1),
    ("BIAS_W0_01", True, "argmin", 2, 0.01),
    ("REFRACTORY", False, "refractory", 2, 1.0),
]
ARM_NAMES = [a[0] for a in ARMS]

SEEDS: List[int] = [42, 7, 13, 100, 200]  # matches V3-EXQ-943 for external comparability

# Substrate properties held constant across all arms (not part of the
# manipulation) -- identical to V3-EXQ-943.
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5  # episodes per block; block parity -> context label
MAX_HARM_BUF = 4000

NUM_SLOTS = 16  # ContextMemory default (E1DeepPredictor constructs it with num_slots=16)

# P0 readiness floors.
WRITE_CALLS_FLOOR = 200.0             # V3-EXQ-943's own floor; that run saw 2900-3200/cell
POSITIVE_CONTROL_Z_FLOOR = 5.0        # instrument must clearly detect a real signal
NEGATIVE_CONTROL_Z_CEILING = 2.0      # instrument must not flag a pure clock as informative

# C-series load-bearing criterion (pre-registered HERE -- not inherited from
# substrate_queue's registered occupancy floor, which is a different question).
N_NULL_PERMUTATIONS = 500
Z_NULL_EXCEED_THRESHOLD = 2.0
C_N_SEEDS_REQUIRED = 3

# Synthetic P0 instrument-validation schedule (no agent, no environment --
# structurally similar magnitude to a real cell's ~3000 writes over 40 blocks,
# per V3-EXQ-943's observed n_write_calls_per_seed).
SYNTHETIC_N_TICKS = 3000
SYNTHETIC_N_BLOCKS = 40


# ------------------------------------------------------------------ #
# Env / agent helpers (verbatim from V3-EXQ-943, the 436 lineage)          #
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


def _make_agent(env: CausalGridWorldV2, write_usage_balancing: bool,
                 write_selection: str, write_refractory_k: int,
                 write_usage_bias_weight: float) -> REEAgent:
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
        contextmemory_write_usage_balancing=write_usage_balancing,
        contextmemory_write_usage_bias_weight=write_usage_bias_weight,
        contextmemory_write_selection=write_selection,
        contextmemory_write_refractory_k=write_refractory_k,
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
    # 436e/f Adam-drift neutralization (see V3-EXQ-943 module docstring).
    # write() still mutates memory.data directly under its own no_grad block
    # regardless.
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
# re-derivation of the selection expression (V3-EXQ-943 mandate, unchanged).#
# ------------------------------------------------------------------ #

class _WriteContextTracker:
    """Records (slot, block_id) for every write ContextMemory.write() actually
    performed, by polling ContextMemory.last_write_index / .slot_write_counts
    (the authoritative instrumentation, maintained in every selection mode
    including legacy) after every agent.sense() call -- identical polling
    contract to V3-EXQ-943's `_WriteSequenceTracker` (see that script for the
    exactness argument: the SD-016 Part B2 hook fires at most one write() per
    sense() call, so polling once per sense() is exact, never approximate).

    block_id (not a raw context label) is recorded so the order-only null
    (`_order_only_null_mis` below) can reassign context labels to BLOCKS while
    holding the exact tick-to-block mapping -- the block structure and its
    autocorrelation -- fixed; only the labeling of which blocks are safe vs
    dangerous is permuted.
    """

    def __init__(self) -> None:
        self.sequence: List[int] = []
        self.block_ids: List[int] = []
        self._last_total = 0

    def poll(self, context_memory, block_id: int) -> None:
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
                self.block_ids.append(block_id)
            self._last_total = total


def _compute_deterministic_dvs(sequence: List[int], num_slots: int) -> Dict[str, Any]:
    """The four reference columns from V3-EXQ-943 (kept for comparability;
    NOT load-bearing here -- see module docstring). Verbatim logic.
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
# Mutual information + order-only null (the new instrument)                #
# ------------------------------------------------------------------ #

def _mutual_information_bits(slot_seq: List[int], context_seq: List[int]) -> float:
    """Exact I(slot; context) in bits from the (slot, context) contingency
    table -- not an estimator, computed directly from counts.
    """
    n = len(slot_seq)
    if n == 0:
        return 0.0
    joint = Counter(zip(slot_seq, context_seq))
    p_slot = Counter(slot_seq)
    p_ctx = Counter(context_seq)
    mi = 0.0
    for (s, c), cnt in joint.items():
        p_sc = cnt / n
        p_s = p_slot[s] / n
        p_c = p_ctx[c] / n
        mi += p_sc * math.log2(p_sc / (p_s * p_c))
    return mi


def _null_seed(label: str, seed: int) -> int:
    """Deterministic, PYTHONHASHSEED-independent seed for the order-only
    null's dedicated random.Random -- separate from torch/numpy/python RNG
    used by training (arm_cell resets those; this runs post-training, purely
    over the already-recorded write sequence, and must not perturb or depend
    on training RNG state).
    """
    h = hashlib.sha256(f"{label}:{seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _order_only_null_mis(
    slot_seq: List[int], block_ids: List[int], n_perm: int, rng_seed: int,
) -> List[float]:
    """Order-only null distribution: the observed write SEQUENCE (slot_seq)
    held fixed, block STRUCTURE (which tick belongs to which block) held
    fixed, and only the assignment of context labels TO blocks permuted.
    A purely order-driven address (a fixed function of tick index -- a
    clock, a period-(k+1) cycle) is therefore statistically indistinguishable
    from its own null by construction; only an address that is a function of
    the TRUE context labels specifically can show excess MI over this null.

    Returns [] (no null; caller must guard) if fewer than 2 distinct blocks
    are present -- a single block carries no context variation to permute
    (this occurs only in a --dry-run smoke, never in a real 200-episode /
    5-episode-per-block run, which spans 40 blocks).
    """
    if not slot_seq:
        return []
    n_blocks = max(block_ids) + 1 if block_ids else 0
    if n_blocks < 2:
        return []
    true_block_ctx = [b % 2 for b in range(n_blocks)]
    rng = random.Random(rng_seed)
    null_mis: List[float] = []
    for _ in range(n_perm):
        perm_block_ctx = true_block_ctx[:]
        rng.shuffle(perm_block_ctx)
        perm_context_seq = [perm_block_ctx[b] for b in block_ids]
        null_mis.append(_mutual_information_bits(slot_seq, perm_context_seq))
    return null_mis


def _z_vs_null(observed_mi: float, null_mis: List[float]) -> Dict[str, Any]:
    if not null_mis:
        return {
            "null_mean_mi_bits": None,
            "null_std_mi_bits": None,
            "null_percentile_rank": None,
            "z_vs_null": None,
            "exceeds_null": False,
            "note": "insufficient_blocks_for_null",
        }
    n = len(null_mis)
    mean = sum(null_mis) / n
    var = sum((x - mean) ** 2 for x in null_mis) / n
    std = math.sqrt(var)
    std_floor = max(std, 1e-9)
    z = (observed_mi - mean) / std_floor
    pct_rank = sum(1 for x in null_mis if x <= observed_mi) / n
    return {
        "null_mean_mi_bits": mean,
        "null_std_mi_bits": std,
        "null_percentile_rank": pct_rank,
        "z_vs_null": z,
        "exceeds_null": z >= Z_NULL_EXCEED_THRESHOLD,
    }


# ------------------------------------------------------------------ #
# P0 instrument validation (synthetic controls, no agent/environment)      #
# ------------------------------------------------------------------ #

def _synthetic_block_ids(n_ticks: int, n_blocks: int) -> List[int]:
    base = n_ticks // n_blocks
    extra = n_ticks % n_blocks
    out: List[int] = []
    for b in range(n_blocks):
        size = base + (1 if b < extra else 0)
        out.extend([b] * size)
    return out


def _validate_mi_instrument() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Positive/negative control checks for the MI-vs-order-only-null
    instrument itself, per the module docstring's "MEASUREMENT INSTRUMENT
    VALIDATION" section. Pure arithmetic -- no agent, no environment.
    """
    block_ids = _synthetic_block_ids(SYNTHETIC_N_TICKS, SYNTHETIC_N_BLOCKS)
    true_context = [b % 2 for b in block_ids]

    # Positive control: address is a PURE function of the true context label
    # (slot 0 if safe, slot 1 if dangerous) -- zero order-per-se content.
    pos_slots = [0 if c == 0 else 1 for c in true_context]
    pos_mi = _mutual_information_bits(pos_slots, true_context)
    pos_null = _order_only_null_mis(
        pos_slots, block_ids, N_NULL_PERMUTATIONS, _null_seed("P0_POSITIVE", 0),
    )
    pos_stats = _z_vs_null(pos_mi, pos_null)
    pos_precondition = {
        "name": "mi_null_test_detects_positive_control",
        "measured": float(pos_stats["z_vs_null"]),
        "threshold": POSITIVE_CONTROL_Z_FLOOR,
        "direction": "lower",
        "control": (
            "synthetic address = f(true context) exactly, zero order-per-se "
            "content; instrument must clearly detect this as informative"
        ),
        "met": pos_stats["z_vs_null"] >= POSITIVE_CONTROL_Z_FLOOR,
        "observed_mi_bits": pos_mi,
        "null_mean_mi_bits": pos_stats["null_mean_mi_bits"],
        "null_std_mi_bits": pos_stats["null_std_mi_bits"],
    }

    # Negative control: address is a PURE function of tick index (round-robin,
    # period=NUM_SLOTS) -- zero context content, indistinguishable in kind
    # from BIAS/REFRACTORY's own order-driven signature.
    neg_slots = [i % NUM_SLOTS for i in range(SYNTHETIC_N_TICKS)]
    neg_mi = _mutual_information_bits(neg_slots, true_context)
    neg_null = _order_only_null_mis(
        neg_slots, block_ids, N_NULL_PERMUTATIONS, _null_seed("P0_NEGATIVE", 0),
    )
    neg_stats = _z_vs_null(neg_mi, neg_null)
    neg_precondition = {
        "name": "mi_null_test_rejects_negative_control",
        "measured": float(neg_stats["z_vs_null"]),
        "threshold": NEGATIVE_CONTROL_Z_CEILING,
        "direction": "upper",
        "control": (
            "synthetic address = tick_index mod NUM_SLOTS exactly, zero "
            "context content; instrument must NOT flag this pure clock as "
            "context-informative (the six-generation artifact-as-verdict "
            "failure mode this null exists to prevent)"
        ),
        "met": neg_stats["z_vs_null"] <= NEGATIVE_CONTROL_Z_CEILING,
        "observed_mi_bits": neg_mi,
        "null_mean_mi_bits": neg_stats["null_mean_mi_bits"],
        "null_std_mi_bits": neg_stats["null_std_mi_bits"],
    }

    return pos_precondition, neg_precondition


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
    tracker: _WriteContextTracker,
    block_id: int,
) -> None:
    """Run a single training episode: sense -> baseline action selection ->
    step -> prediction-loss training -> harm_eval training. No sleep, no
    context-conditioned action selection, no SD-016 arming -- identical
    isolation to V3-EXQ-943.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
        agent.clock.advance()
        tracker.poll(agent.e1.context_memory, block_id)  # exact -- see class docstring

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


def _run_cell(arm_name: str, write_usage_balancing: bool, write_selection: str,
              write_refractory_k: int, write_usage_bias_weight: float, seed: int,
              base_config_slice: Dict[str, Any], zg: ZGoalStreamAccumulator,
              training_episodes: int, steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "arm": arm_name,
        "contextmemory_write_usage_balancing": write_usage_balancing,
        "contextmemory_write_usage_bias_weight": write_usage_bias_weight,
        "contextmemory_write_selection": write_selection,
        "contextmemory_write_refractory_k": write_refractory_k,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(
            env_safe, write_usage_balancing, write_selection, write_refractory_k,
            write_usage_bias_weight,
        )

        standard_params = [
            p for n, p in agent.named_parameters()
            if "harm_eval_head" not in n and "context_memory.memory" not in n
        ]
        harm_eval_params = list(agent.e3.harm_eval_head.parameters())
        optimizer = optim.Adam(standard_params, lr=1e-3)
        harm_eval_opt = optim.Adam(harm_eval_params, lr=1e-4)

        harm_buf_pos: List[torch.Tensor] = []
        harm_buf_neg: List[torch.Tensor] = []
        tracker = _WriteContextTracker()

        agent.train()
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            _run_episode(
                agent, env, steps_per_episode, optimizer, harm_eval_opt,
                harm_buf_pos, harm_buf_neg, tracker, block,
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

        context_seq = [b % 2 for b in tracker.block_ids]
        observed_mi = _mutual_information_bits(tracker.sequence, context_seq)
        null_mis = _order_only_null_mis(
            tracker.sequence, tracker.block_ids, N_NULL_PERMUTATIONS,
            _null_seed(arm_name, seed),
        )
        null_stats = _z_vs_null(observed_mi, null_mis)

        row: Dict[str, Any] = {
            "arm": arm_name,
            "seed": seed,
            "write_usage_balancing": write_usage_balancing,
            "write_usage_bias_weight": write_usage_bias_weight,
            "write_selection": write_selection,
            "write_refractory_k": write_refractory_k,
            **dvs,
            "n_blocks_observed": (max(tracker.block_ids) + 1) if tracker.block_ids else 0,
            "observed_mi_bits": observed_mi,
            **null_stats,
        }
        cell.stamp(row)

    print(
        f"verdict: {'PASS' if row['exceeds_null'] else 'FAIL'}",
        flush=True,
    )
    return row


# ------------------------------------------------------------------ #
# Top-level run                                                            #
# ------------------------------------------------------------------ #

def run(dry_run: bool = False) -> Tuple[Dict[str, Any], ZGoalStreamAccumulator]:
    zg = ZGoalStreamAccumulator()

    training_episodes = 10 if dry_run else TRAINING_EPISODES  # >=2 blocks for a non-degenerate smoke null
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
    }

    # --- P0a: instrument validation (synthetic, no agent -- runs first, cheap) ---
    pos_precondition, neg_precondition = _validate_mi_instrument()
    if not (pos_precondition["met"] and neg_precondition["met"]):
        label = "mi_instrument_not_validated"
        result: Dict[str, Any] = {
            "outcome": "FAIL",
            "status": "FAIL",
            "claim_ids": CLAIM_IDS,
            "experiment_type": EXPERIMENT_TYPE,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "evidence_direction": "non_contributory",
            "sleep_driver_pattern": "N/A (no sleep loop)",
            "arm_results": [],
            "interpretation": {
                "label": label,
                "preconditions": [pos_precondition, neg_precondition],
                "criteria_non_degenerate": {},
            },
            "fatal_error_count": 0,
        }
        return result, zg

    arm_results: List[Dict[str, Any]] = []
    for arm_name, wub, wsel, wrk, wbw in ARMS:
        for seed in seeds:
            row = _run_cell(
                arm_name, wub, wsel, wrk, wbw, seed, full_config_slice, zg,
                training_episodes, steps_per_episode,
            )
            arm_results.append(row)

    by_arm: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ARM_NAMES}
    for row in arm_results:
        by_arm[row["arm"]].append(row)

    # --- P0b: writepath genuinely engaged in every cell ---
    write_call_counts = [row["n_write_calls"] for row in arm_results]
    min_write_calls = min(write_call_counts) if write_call_counts else 0.0
    try:
        preconditions = p0_readiness_gate([
            pos_precondition,
            neg_precondition,
            {
                "name": "writepath_engaged_every_cell",
                "measured": float(min_write_calls),
                "threshold": WRITE_CALLS_FLOOR,
                "direction": "lower",
            },
        ])
    except P0NotReady as e:
        label = "substrate_not_ready_requeue"
        result = {
            "outcome": "FAIL",
            "status": "FAIL",
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

    # --- C: load-bearing criterion, per arm ---
    def _seeds_clearing_null(arm_name: str) -> int:
        return sum(1 for row in by_arm[arm_name] if row["exceeds_null"])

    n_blocks_min = min(
        (row["n_blocks_observed"] for row in arm_results), default=0,
    )

    criteria = []
    per_arm_pass: Dict[str, bool] = {}
    for arm_name in ARM_NAMES:
        n_clear = _seeds_clearing_null(arm_name)
        arm_pass = n_clear >= C_N_SEEDS_REQUIRED
        per_arm_pass[arm_name] = arm_pass
        criteria.append({
            "name": f"C_{arm_name}_exceeds_order_only_null",
            "load_bearing": True,
            "passed": arm_pass,
            "n_seeds_clearing_null": n_clear,
            "n_seeds_required": C_N_SEEDS_REQUIRED,
            "z_threshold": Z_NULL_EXCEED_THRESHOLD,
        })

    criteria_non_degenerate = {
        f"C_{arm_name}_exceeds_order_only_null": (len(seeds) >= 2 and n_blocks_min >= 2)
        for arm_name in ARM_NAMES
    }

    overall_pass = any(per_arm_pass.values())
    status = "PASS" if overall_pass else "FAIL"
    if overall_pass:
        clearing_arms = [a for a, p in per_arm_pass.items() if p]
        label = "context_informative_address_found_at_operating_point"
    else:
        clearing_arms = []
        label = "no_operating_point_exceeds_order_only_null"

    evidence_direction = "non_contributory"  # claim_ids=[]; readiness/context only

    metrics: Dict[str, Any] = {}
    for arm_name in ARM_NAMES:
        rows = by_arm[arm_name]
        metrics[f"n_occupied_slots_per_seed_{arm_name}"] = [r["n_occupied_slots"] for r in rows]
        metrics[f"n_write_calls_per_seed_{arm_name}"] = [r["n_write_calls"] for r in rows]
        metrics[f"entropy_bits_per_seed_{arm_name}"] = [r["entropy_bits"] for r in rows]
        metrics[f"self_repeat_rate_per_seed_{arm_name}"] = [r["self_repeat_rate"] for r in rows]
        metrics[f"round_robin_agreement_per_seed_{arm_name}"] = [r["round_robin_agreement"] for r in rows]
        metrics[f"observed_mi_bits_per_seed_{arm_name}"] = [r["observed_mi_bits"] for r in rows]
        metrics[f"z_vs_null_per_seed_{arm_name}"] = [r["z_vs_null"] for r in rows]
        metrics[f"n_seeds_clearing_null_{arm_name}"] = _seeds_clearing_null(arm_name)

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory write-ADDRESS informativeness diagnostic\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; informs EVB-0628/INV-044 "
        f"substrate readiness, contextmemory-write-path-addressing-degeneracy)\n\n"
        f"| Arm | seeds clearing null (z>={Z_NULL_EXCEED_THRESHOLD}) | observed MI range (bits) |\n"
        f"|---|---|---|\n"
        + "".join(
            f"| {arm_name} | {_seeds_clearing_null(arm_name)}/{len(by_arm[arm_name])} "
            f"| {min(r['observed_mi_bits'] for r in by_arm[arm_name]):.4f}-"
            f"{max(r['observed_mi_bits'] for r in by_arm[arm_name]):.4f} |\n"
            for arm_name in ARM_NAMES
        )
        + f"\nClearing arms: {clearing_arms if clearing_arms else '(none)'}\n"
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
            "criteria_aggregation": "any",
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
        "write_calls_floor": WRITE_CALLS_FLOOR,
        "z_null_exceed_threshold": Z_NULL_EXCEED_THRESHOLD,
        "c_n_seeds_required": C_N_SEEDS_REQUIRED,
        "n_null_permutations": N_NULL_PERMUTATIONS,
        "positive_control_z_floor": POSITIVE_CONTROL_Z_FLOOR,
        "negative_control_z_ceiling": NEGATIVE_CONTROL_Z_CEILING,
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
