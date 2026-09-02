#!/opt/local/bin/python3
"""
V3-EXQ-972 -- ContextMemory write-content discrimination: H4 (input-distribution)
representation-structure probe, real-agent, against the frozen hypothesis-space
question `contextmemory_write_content_discrimination` (registered from the
V3-EXQ-956 autopsy).

Chip: chip-20260830-ctxmem-write-content-h1h4-portfolio

experiment_purpose: diagnostic

hypothesis_space:
  qid: contextmemory_write_content_discrimination
  hid: H4-input-distribution

H4 (red-team addition): "Train-time z_world under-differentiation (SD-008
~0.98-cosine cone) leaves no usable content structure for ANY write-side
objective." Null (this experiment's own falsifier, verbatim): "latents carry
separable structure the tagger ignores" -- i.e. if the ACTUAL write-stream
states already show clear inter-context separation via a simple
representation-only statistic, that REFUTES H4 and refocuses blame on H1/H3
(the addressing ALGORITHM, not the input it receives).

WHY THIS RUN EXISTS. V3-EXQ-956 trained `write_selection="gumbel_learned"`
with `contextmemory_write_addressing_loss_weight=0.5` under a real agent and
measured whether the trained `write_addr_tagger` becomes content-conditioned
(post-training synthetic 2-cluster Jaccard probe). It did NOT instrument what
the tagger actually RECEIVES during training -- the real write-stream state
tensor, [z_self, z_world] concatenated, as it flows through 15,000 real
training steps. 956's own docstring says this explicitly: the content-
discrimination question is measured only at the tagger's OUTPUT (post-hoc,
eval-mode, on a synthetic stream), never at its INPUT (train-time, on the
real agent's own latents). This experiment closes that gap: it runs the SAME
GUMBEL_TRAINED operating point (write_selection="gumbel_learned",
contextmemory_write_addressing_loss_weight=0.5) and, in addition, records the
REAL train-time write-stream state at every step, bucketed by which context
(safe / dangerous) produced it, and measures whether those two buckets are
representationally separable by a simple statistic (pairwise cosine
similarity) that has NOTHING to do with write_addr_tagger's own learned
scoring function.

This experiment makes NO claim about whether GUMBEL_TRAINED's addressing
training succeeds (956/H1 already own that question) and is ORTHOGONAL to
H1/H2/H3's own algorithm-side experiments. It measures ONE thing:
representation structure at the write-stream input, independent of what any
downstream write-selection algorithm does with it.

DESIGN (single arm, no new loss -- this is instrumentation-only; the training
objective is IDENTICAL to V3-EXQ-956's own GUMBEL_TRAINED arm):
  GUMBEL_TRAINED   write_selection="gumbel_learned",
                    contextmemory_write_addressing_loss_weight=0.5   (956's
                    own operating point; training happens exactly as in
                    956's GUMBEL_TRAINED arm -- this experiment does not test
                    whether addressing training succeeds, it instruments the
                    INPUT that training receives regardless of outcome).

No LEGACY / GUMBEL_UNTRAINED comparator arms: 956 already owns the
occupancy/mechanical-control comparison across those three arms. Re-running
them here would duplicate 956's own measurement rather than add the new one.

SEEDS: [42, 7, 13, 100, 200] -- identical to V3-EXQ-956 / V3-EXQ-943, for
direct read-across comparability (same env construction, same training
schedule, same RNG-consuming call sequence up to the new instrumentation,
which only READS latents that already exist -- it adds no extra RNG draws).

SCHEDULE: TRAINING_EPISODES=100, STEPS_PER_EPISODE=150 -- 956's own full,
non-toy schedule (matches V3-EXQ-907's real, non-toy precedent). Chosen over
a moderate-scale shortcut because representation structure is a property
worth measuring at real training scale, not a cheap approximation of it.

Step 2.4 GOV-REUSE-1 (existing-evidence check): V3-EXQ-956's own manifest
cells are NOT reusable for this measurement, for two independent reasons.
(1) 956 never recorded the per-step write-stream state tensors this
experiment's DV depends on -- 956's `arm_results` rows carry only the
deterministic write-sequence DVs (`_compute_deterministic_dvs`) and the
post-training synthetic-probe Jaccard, neither of which is a substitute for
the actual train-time [z_self, z_world] distribution. (2) Even setting that
aside, 956's GUMBEL_TRAINED cells were minted WITHOUT
`include_driver_script_in_hash=False` (956 never opted into cross-driver
baseline reuse -- it is not the first-of-lineage baseline-factoring
experiment), so a different driver (this one) is refused by the arm-fingerprint
mechanism even for whatever 956 DID record. Neither obstacle is recoverable
post-hoc; proceeding to run fresh.

INSTRUMENTATION: at every training step, immediately after
`latent = agent.sense(...)`, the write-stream state tensor is reconstructed
EXACTLY as `ree_core/agent.py`'s own `context_memory.write()` call site
constructs it (confirmed by grep -- see the smoke-test report for the exact
call-site text): `torch.cat([latent.z_self.detach(), latent.z_world.detach()],
dim=-1)`. This driver's `_make_agent` never sets `use_coalition_controller`
(REEConfig default False, unchanged from 956), so `agent.coalition` stays
None throughout and the SD-091/MECH-481 `write_gate()` scaling at that call
site is the identity -- the reconstructed tensor here is bit-identical to
what `ContextMemory.write()` actually receives, with no approximation. The
tensor is appended (`.detach().cpu()`, no autograd graph retained) into one
of two per-cell rolling buffers keyed by which environment (safe/dangerous)
produced the observation this episode, capped at MAX_STATE_BUF (4000)
entries via the same `del buf[:-MAX_BUF]` eviction idiom this driver's own
`harm_buf_pos`/`harm_buf_neg` buffers use (itself copied from 956/943's own
convention), so memory never grows unbounded across a 15,000-step run.

MEASUREMENT (post-training, per seed, from the two buffers):
  intra_safe_cosine       mean pairwise cosine similarity within states_safe
  intra_dangerous_cosine  mean pairwise cosine similarity within states_dangerous
  inter_cosine            mean pairwise cosine similarity, EVERY states_safe
                           sample against EVERY states_dangerous sample
  separability_score = ((intra_safe_cosine + intra_dangerous_cosine) / 2.0)
                        - inter_cosine
    Kept in SIMILARITY units (not distance), so a POSITIVE score means
    same-context states are more similar to each other than to
    opposite-context states -- the intuitive "separable" direction. Computed
    via `torch.nn.functional.normalize` + matmul, no external libraries.
  Both buffers are subsampled to at most PROBE_SUBSAMPLE_MAX (500) states via
  `torch.randperm` before the pairwise computation, for O(n^2) cost control.

cross_cluster_cone_check (the SD-008 "~0.98-cosine cone" reference H4's null
cites): `intra_safe_cosine` and `intra_dangerous_cosine` are reported
directly (not just folded into separability_score) so a reader can see
whether the WHOLE representation -- regardless of context -- sits in a tight
near-1.0 cosine cone (would explain low separability structurally, i.e. H4's
own proposed mechanism) vs. genuinely spread out.

CROSS-REFERENCE, non-gating: this run's own trained `write_addr_tagger` is
also probed post-training with 956's own eval-mode 2-cluster synthetic
Jaccard instrument (`_run_content_probe`, copied verbatim from 956), so a
reader can directly compare "was the real-agent representation separable"
(this experiment's own load-bearing measurement) against "did the tagger's
own synthetic-probe content-discrimination succeed" (956/H1's own load-bearing
question) for the SAME seeds. Nothing is gated on this comparison -- 956/H1
already own the load-bearing test of whether addressing training succeeds.

ACCEPTANCE CRITERIA -- this is a MEASUREMENT diagnostic, not a pass/fail on a
scientific direction:

P0 READINESS (writepath engagement, exactly 956's own floor and gate shape):
  min(n_write_calls across seeds) >= WRITE_CALLS_FLOOR (200). On failure,
  self-routes to `substrate_not_ready_requeue`, exactly as 956.

H4 READINESS (separate from P0 -- gates the separability measurement itself,
not writepath engagement generally): per seed,
  min(len(states_safe), len(states_dangerous)) >= H4_SEED_READINESS_FLOOR (50)
  -- enough samples in BOTH buckets for a meaningful pairwise-cosine estimate.
If unmet on a given seed, that seed is EXCLUDED from the aggregate (not
treated as a hard failure by itself) and the exclusion is recorded per-seed.
If unmet on EVERY seed, the run self-routes to a DISTINCT label,
`h4_insufficient_write_samples` -- deliberately NOT `substrate_not_ready_requeue`,
since this is not a substrate-readiness issue (P0 already confirmed the
writepath fires); the buffers are simply too small for this specific
measurement, a distinct failure mode from writepath non-engagement.

LOAD-BEARING CRITERION `H4_representation_separability_measured`
(load_bearing: true, kind: "measurement"): PASSES (is non-degenerate) iff the
H4 readiness precondition above held on >= H4_N_SEEDS_REQUIRED (3) of the 5
seeds -- REGARDLESS of what separability_score's sign turns out to be. This
criterion certifies that the measurement was taken meaningfully, not that
representations are or are not separable; do not read `passed: true` here as
"H4 refuted" or `passed: false` as "H4 supported" -- see the separate
descriptive block below for that reading.

DESCRIPTIVE (non-gating) finding: `interpretation.h4_reading` reports which
of two readings the mean `separability_score` (across seeds meeting H4
readiness) supports:
  "representation_separable"        mean separability_score > H4_NOISE_FLOOR
                                     (0.05) -- REFUTES H4 as the write-content
                                     blocker; points back to H1/H3 (algorithm).
  "representation_undifferentiated" mean separability_score <= H4_NOISE_FLOOR
                                     -- SUPPORTS H4.
H4_NOISE_FLOOR = 0.05 is stated explicitly as a CONSERVATIVE noise floor for
a cosine-similarity statistic, not a rigorously derived value -- this is
recorded plainly rather than overclaiming precision the measurement does not
have.

criteria_non_degenerate: {"H4_representation_separability_measured": <bool>}
  True iff the H4 readiness precondition held on >= 3/5 seeds (same
  condition the load-bearing criterion's `passed` reads).

evidence_direction: "non_contributory". claim_ids: [] -- this experiment
measures write-stream REPRESENTATION STRUCTURE (a substrate-diagnostic
property), not a claim hypothesis; per this file's own convention (matching
956/943), diagnostic-purpose experiments with claim_ids=[] are excluded from
governance confidence scoring by design.

SUBSTRATE PROPERTIES held constant, identical to V3-EXQ-956's own list (reused
verbatim so the two experiments' training dynamics are the same measurement
up to RNG and this driver's added, read-only instrumentation):
  - contextmemory_gated_content_write=True (436d write-path repair).
  - sd016_writepath_mode="sense_only".
  - alpha_world=0.9 (SD-008).
  - use_noise_floor=True (MECH-313/ARC-065).
  - context_memory.memory.requires_grad_(False) after construction (436e/f
    Adam-drift neutralization). write_addr_tagger is left trainable.
  - use_per_stream_vs=True, use_anchor_sets=True, use_sd039_anchor_payload=True.
  - use_salience_coordinator / use_coalition_controller left at REEConfig
    defaults (both False) -- see Step 2.5c below and the INSTRUMENTATION
    section above for why this matters to the state-reconstruction claim.
  - sd016_enabled left at its REEConfig default (False).

No sleep. Identical rationale to V3-EXQ-956/943: SD-016 sleep-cycle
consolidation is orthogonal to whether the write-ADDRESS input representation
carries usable content structure. sleep_driver_pattern: "N/A (no sleep loop)".

Step 2.5b re-derive brake: N/A. claim_ids is empty (see above) -- there is no
claim-hypothesis re-test here for the brake to apply to.

red-team (fable), 2026-09-02: CONTESTED, one finding, DISMISSED with this
caveat rather than fixed. `separability_score` has no null control for
temporal/episode autocorrelation: intra-bucket pairs include same-episode
and same-CONTEXT_SWITCH_EVERY(5)-episode-block pairs, while inter-bucket
pairs are always cross-block and separated in training time, so slow
weight/representation drift over 100 episodes could produce `intra - inter
> 0` with zero context-CONDITIONED content -- and only scalar stats
(intra/inter cosines, separability_score) reach the manifest, with no
per-state episode/step index, so no permutation/shuffle null can be
computed post-hoc if this is ever suspected. NOT fixed here: the concrete
fix (store (episode_idx, step_idx) alongside each buffered state; add a
block-shuffle permutation control reporting `separability_score_null_mean/
max` per seed) is a real but non-trivial addition to the instrumentation
path, out of scope for this pass. Mitigating factors that make disclosure
(rather than blocking) the right call: (a) this experiment's own load-
bearing criterion (`H4_representation_separability_measured`) already
gates only on sample-sufficiency, not on the sign of separability_score --
the scientific reading (`interpretation.h4_reading`) is explicitly
non-gating and descriptive; (b) `intra_safe_cosine`/`intra_dangerous_
cosine` are reported separately (not just folded into the difference), so
a reader CAN see whether the whole representation sits in a uniformly
tight cone (autocorrelation-consistent) versus genuinely separated; (c)
a "representation_separable" reading here is cross-checked, non-gating,
against 956's own trained-tagger Jaccard probe on the SAME seeds, which
uses a synthetic, non-temporally-autocorrelated stream and is not subject
to this confound -- persistent disagreement between the two would itself
be a flag to a governance reviewer that the real-agent reading needs the
permutation-null treatment before being taken at face value.

Step 2.5c substrate-path overlap (re-checked against the CURRENT
substrate_queue.json, 2026-09-01, NOT merely cited from V3-EXQ-956): every
open `corrupting` entry whose `substrate_paths` names `ree_core/agent.py` or
`ree_core/predictors/e1_deep.py` was re-read fresh:
  - `contextmemory-write-path-addressing-degeneracy` (e1_deep.py::ContextMemory.write,
    status implemented_pending_validation) -- this IS the entry this
    experiment's write-stream is drawn from; not an unrelated overlap.
  - `modulatory-bias-selection-authority` (agent.py::REEAgent.select_action,
    status implemented) -- closed/non-blocking.
  - `mode-governance-engagement` (agent.py, whole-file,
    status implemented_pending_validation) -- removed by construction:
    this driver never sets `use_salience_coordinator=True` (REEConfig's own
    False default is used throughout, identical to 956) and never reads
    `agent.operating_mode` or any SalienceCoordinator output.
  - `SD-082` (agent.py, whole-file, status implemented_pending_validation) --
    removed by construction: `lateral_pfc_rule_readout_consumer` is a no-op
    default per its own implementation_note_update ("OFF bit-identical");
    this driver never enables the SD-078 rule-pool / SD-082 consumer path.
  - `SD-e1-rollout-consistency-training` (e1_deep.py::forward,
    e1_deep.py::predict_long_horizon, status item1_validated_item2_pending)
    -- removed by construction: this driver never calls
    `predict_long_horizon` (no multi-step rollout) and never enables an
    action-conditioned / rollout-consistency training objective; it only
    exercises the ordinary single-step `agent.sense()` -> E1 forward path,
    which is unaffected by this entry's still-pending ITEM 2.
No other open `corrupting` entry's substrate_paths overlaps a module this
driver imports or exercises. `SD-056` no longer even names agent.py in the
current queue (e2_fast.py / e3_selector.py / config.py only) -- non-blocking
either way.

Does NOT flip the `contextmemory-write-path-addressing-degeneracy`
substrate_queue entry's status regardless of outcome -- that stays a
human/governance disposition, unchanged from 956's own convention.

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


EXPERIMENT_TYPE = "v3_exq_972_contextmemory_write_content_h4_input_distribution"
QUEUE_ID = "V3-EXQ-972"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# --- Single arm (see module docstring "DESIGN") --- #
ARM_NAME = "GUMBEL_TRAINED"
WRITE_SELECTION = "gumbel_learned"
WRITE_ADDRESSING_LOSS_WEIGHT = 0.5

SEEDS: List[int] = [42, 7, 13, 100, 200]  # matches V3-EXQ-956 / V3-EXQ-943

# Substrate properties held constant (not the manipulation) -- identical to
# V3-EXQ-956's own list.
CONTEXTMEMORY_GATED_CONTENT_WRITE = True
SD016_WRITEPATH_MODE = "sense_only"
ALPHA_WORLD = 0.9
USE_NOISE_FLOOR = True
NOISE_FLOOR_ALPHA = 0.1
NOISE_FLOOR_MIN_TEMPERATURE = 1.0
BASELINE_TEMPERATURE = 1.0

TRAINING_EPISODES = 100  # matches V3-EXQ-956/907's own real (non-toy) schedule
STEPS_PER_EPISODE = 150
CONTEXT_SWITCH_EVERY = 5
MAX_HARM_BUF = 4000
MAX_STATE_BUF = 4000  # per-context write-stream state buffer cap

NUM_SLOTS = 16  # ContextMemory default
LATENT_DIM = 64  # self_dim(32) + world_dim(32) -- must match the contract
                 # test's own LATENT_DIM for the cross-reference probe to be
                 # a like-for-like measurement.

# P0 readiness floor -- exactly V3-EXQ-956/943's own.
WRITE_CALLS_FLOOR = 200.0

# H4 readiness: per-seed minimum sample count in EACH context bucket for the
# pairwise-cosine separability measurement to be meaningful.
H4_SEED_READINESS_FLOOR = 50
H4_N_SEEDS_REQUIRED = 3  # >= 3/5 seeds required for the load-bearing criterion

# Conservative noise floor for reading separability_score's sign -- stated
# explicitly as a floor choice, not a rigorously derived value (see module
# docstring).
H4_NOISE_FLOOR = 0.05

# Subsample cap for the O(n^2) pairwise-cosine computation.
SEP_SUBSAMPLE_MAX = 500

# Cross-reference post-training synthetic 2-cluster probe (956's own
# instrument -- test_contextmemory_write_address_selection.py's _stream).
PROBE_N = 1500
PROBE_JITTER = 0.0078
PROBE_CLUSTERS = 2

TAGGER_MOVE_EPS = 1e-8  # threshold for "did write_addr_tagger's state_dict move"


# ------------------------------------------------------------------ #
# Env / agent helpers (env params reused verbatim from V3-EXQ-956/943)     #
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
    assert agent.coalition is None, (
        "agent.coalition is not None despite use_coalition_controller "
        "defaulting False -- the write-stream state reconstruction in this "
        "driver assumes the SD-091/MECH-481 write_gate() scaling at "
        "context_memory.write()'s call site is the identity; update the "
        "reconstruction (or this assertion) before trusting the "
        "separability measurement if this ever fires."
    )
    assert agent.e1.context_memory.num_slots == NUM_SLOTS, (
        f"ContextMemory num_slots changed from the assumed {NUM_SLOTS} -- "
        "update NUM_SLOTS before trusting occupancy-floor thresholds."
    )
    total_latent_dim = agent.e1.config.self_dim + agent.e1.config.world_dim
    assert total_latent_dim == LATENT_DIM, (
        f"self_dim+world_dim={total_latent_dim} != LATENT_DIM={LATENT_DIM} -- "
        "the cross-reference synthetic probe would no longer match the "
        "contract test's own instrument; update LATENT_DIM before trusting "
        "the cross-reference measurement."
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
    # write_addr_tagger is deliberately NOT frozen -- it is the mechanism's
    # own trainable parameter (irrelevant to THIS experiment's own DV, which
    # reads the write-stream INPUT, not the tagger's output, but left
    # trainable to keep training dynamics identical to 956's GUMBEL_TRAINED
    # arm).
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
    temperature. Reused verbatim from the 436/943/956 lineage's baseline
    policy.
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
# re-derivation of the selection expression (architecture doc mandate,     #
# identical to V3-EXQ-956).                                                #
# ------------------------------------------------------------------ #

class _WriteSequenceTracker:
    """Records the ordered sequence of slots ContextMemory.write() actually
    mutated, by polling ContextMemory.last_write_index / .slot_write_counts
    (the authoritative instrumentation, maintained in every selection mode
    including gumbel_learned) after every agent.sense() call -- never by
    recomputing write()'s own scoring expression. Identical to V3-EXQ-956's
    tracker.
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
    """The deterministic occupancy-family columns V3-EXQ-943/956 established
    as robust -- reused for P0 readiness (writepath engagement) only; this
    experiment's own load-bearing DV is the separability measurement below.
    """
    n_write_calls = len(sequence)
    if n_write_calls == 0:
        return {
            "n_write_calls": 0,
            "n_occupied_slots": 0,
            "entropy_bits": 0.0,
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

    return {
        "n_write_calls": n_write_calls,
        "n_occupied_slots": n_occupied,
        "entropy_bits": entropy,
    }


# ------------------------------------------------------------------ #
# H4 representation-separability measurement (this experiment's OWN,       #
# load-bearing DV -- see module docstring "MEASUREMENT").                  #
# ------------------------------------------------------------------ #

def _compute_separability_stats(
    states_safe: List[torch.Tensor],
    states_dangerous: List[torch.Tensor],
    max_n: int = SEP_SUBSAMPLE_MAX,
) -> Dict[str, Any]:
    """Representation-only pairwise-cosine separability between the two
    per-context write-stream state buffers. Subsamples each buffer to at
    most `max_n` entries via torch.randperm for O(n^2) cost control.
    Returns None-valued fields if either buffer is empty (caller is expected
    to have already checked the H4 readiness floor before calling this for a
    seed that will contribute to the aggregate; called defensively here too).
    """
    n_safe_raw = len(states_safe)
    n_dang_raw = len(states_dangerous)
    if n_safe_raw == 0 or n_dang_raw == 0:
        return {
            "intra_safe_cosine": None,
            "intra_dangerous_cosine": None,
            "inter_cosine": None,
            "separability_score": None,
            "n_safe_used": 0,
            "n_dangerous_used": 0,
        }

    x_safe = torch.cat(states_safe, dim=0)  # [n_safe_raw, LATENT_DIM]
    x_dang = torch.cat(states_dangerous, dim=0)  # [n_dang_raw, LATENT_DIM]

    if x_safe.shape[0] > max_n:
        idx = torch.randperm(x_safe.shape[0])[:max_n]
        x_safe = x_safe[idx]
    if x_dang.shape[0] > max_n:
        idx = torch.randperm(x_dang.shape[0])[:max_n]
        x_dang = x_dang[idx]

    xs_n = F.normalize(x_safe, dim=-1)
    xd_n = F.normalize(x_dang, dim=-1)

    n_s = xs_n.shape[0]
    n_d = xd_n.shape[0]

    inter_cosine = (xs_n @ xd_n.T).mean().item()

    if n_s >= 2:
        sim_safe = xs_n @ xs_n.T
        mask_s = ~torch.eye(n_s, dtype=torch.bool)
        intra_safe_cosine: Optional[float] = sim_safe[mask_s].mean().item()
    else:
        intra_safe_cosine = None

    if n_d >= 2:
        sim_dang = xd_n @ xd_n.T
        mask_d = ~torch.eye(n_d, dtype=torch.bool)
        intra_dangerous_cosine: Optional[float] = sim_dang[mask_d].mean().item()
    else:
        intra_dangerous_cosine = None

    if intra_safe_cosine is not None and intra_dangerous_cosine is not None:
        separability_score: Optional[float] = (
            (intra_safe_cosine + intra_dangerous_cosine) / 2.0
        ) - inter_cosine
    else:
        separability_score = None

    return {
        "intra_safe_cosine": intra_safe_cosine,
        "intra_dangerous_cosine": intra_dangerous_cosine,
        "inter_cosine": inter_cosine,
        "separability_score": separability_score,
        "n_safe_used": n_s,
        "n_dangerous_used": n_d,
    }


# ------------------------------------------------------------------ #
# Post-training content-discrimination CROSS-REFERENCE probe (copied       #
# verbatim from V3-EXQ-956 -- non-gating, context only; see module         #
# docstring "CROSS-REFERENCE").                                            #
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
    """Post-training, EVAL-MODE probe, copied verbatim from V3-EXQ-956. Uses
    a probe-specific seed offset so this measurement's RNG stream never
    collides with the training seed's own consumption earlier in the same
    cell.
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
    states_safe: List[torch.Tensor],
    states_dangerous: List[torch.Tensor],
    is_safe_ep: bool,
    tracker: _WriteSequenceTracker,
) -> None:
    """Run a single training episode: sense -> E1 tick -> baseline action
    selection -> step -> prediction-loss training -> harm_eval training.
    Reused verbatim from V3-EXQ-956's episode runner (including the
    `agent._e1_tick(latent)` call, mandatory for the write-addressing loss to
    actually reach the backward graph -- see 956's own docstring "DELIBERATE
    DIVERGENCE" section for why), with ONE addition: after computing
    `latent`, the write-stream state tensor is reconstructed exactly as
    `context_memory.write()`'s own caller in ree_core/agent.py constructs it
    and appended to the per-context buffer for THIS episode's context
    (`states_safe` if `is_safe_ep` else `states_dangerous`). This is a
    read-only addition -- no change to the training objective, the action
    policy, or any RNG-consuming call.
    """
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    state_buf = states_safe if is_safe_ep else states_dangerous

    for _step in range(steps):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        obs_harm = obs_dict.get("harm_obs", None)

        latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

        # H4 instrumentation: reconstruct the write-stream state tensor
        # EXACTLY as ree_core/agent.py's context_memory.write() call site
        # does (torch.cat([z_self, z_world], dim=-1).detach()); agent.coalition
        # is asserted None in _make_agent so the write_gate() scaling at that
        # call site is the identity and this reconstruction is exact, not an
        # approximation. .cpu() keeps the buffer off any accelerator device
        # across a long run; no autograd graph is retained.
        obs_state = torch.cat(
            [latent.z_self.detach(), latent.z_world.detach()], dim=-1
        ).cpu()
        state_buf.append(obs_state)
        if len(state_buf) > MAX_STATE_BUF:
            del state_buf[:-MAX_STATE_BUF]

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


def _run_cell(seed: int, base_config_slice: Dict[str, Any],
              zg: ZGoalStreamAccumulator, training_episodes: int,
              steps_per_episode: int) -> Dict[str, Any]:
    cell_config_slice = {
        **base_config_slice,
        "arm": ARM_NAME,
        "contextmemory_write_selection": WRITE_SELECTION,
        "contextmemory_write_addressing_loss_weight": WRITE_ADDRESSING_LOSS_WEIGHT,
    }
    with arm_cell(seed, config_slice=cell_config_slice, script_path=Path(__file__)) as cell:
        env_safe = _make_env_safe(seed)
        env_dang = _make_env_dangerous(seed)
        agent = _make_agent(env_safe, WRITE_SELECTION, WRITE_ADDRESSING_LOSS_WEIGHT)

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

        agent.train()
        print(f"Seed {seed} Condition {ARM_NAME}", flush=True)
        for ep in range(training_episodes):
            block = ep // CONTEXT_SWITCH_EVERY
            is_safe_ep = (block % 2 == 0)
            env = env_safe if is_safe_ep else env_dang
            _run_episode(
                agent, env, steps_per_episode, optimizer, harm_eval_opt,
                harm_buf_pos, harm_buf_neg, states_safe, states_dangerous,
                is_safe_ep, tracker,
            )
            if (ep + 1) % 50 == 0 or (ep + 1) == training_episodes:
                print(
                    f"  [train] arm={ARM_NAME} seed={seed} "
                    f"ep {ep + 1}/{training_episodes} "
                    f"n_writes={len(tracker.sequence)} "
                    f"n_states_safe={len(states_safe)} "
                    f"n_states_dangerous={len(states_dangerous)}",
                    flush=True,
                )

        zg.observe(agent)
        dvs = _compute_deterministic_dvs(tracker.sequence, NUM_SLOTS)

        final_state = agent.e1.context_memory.write_addr_tagger.state_dict()
        max_abs_diff = 0.0
        for k, v0 in tagger_init_state.items():
            d = (final_state[k] - v0).abs().max().item()
            max_abs_diff = max(max_abs_diff, d)
        tagger_params_moved = max_abs_diff > TAGGER_MOVE_EPS

        n_states_safe_raw = len(states_safe)
        n_states_dangerous_raw = len(states_dangerous)
        seed_readiness_met = (
            min(n_states_safe_raw, n_states_dangerous_raw) >= H4_SEED_READINESS_FLOOR
        )
        sep_stats = _compute_separability_stats(states_safe, states_dangerous)

        # Cross-reference only (non-gating) -- see module docstring.
        probe_jaccard, probe_per_cluster = _run_content_probe(agent, seed)

        row: Dict[str, Any] = {
            "arm": ARM_NAME,
            "seed": seed,
            "write_selection": WRITE_SELECTION,
            "contextmemory_write_addressing_loss_weight": WRITE_ADDRESSING_LOSS_WEIGHT,
            **dvs,
            "n_states_safe_raw": n_states_safe_raw,
            "n_states_dangerous_raw": n_states_dangerous_raw,
            "seed_readiness_met": seed_readiness_met,
            **sep_stats,
            "tagger_params_moved": tagger_params_moved,
            "tagger_max_abs_param_diff": max_abs_diff,
            "probe_2cluster_jaccard": probe_jaccard,
            "probe_2cluster_occupied": {
                str(cid): sorted(slots) for cid, slots in probe_per_cluster.items()
            },
        }
        cell.stamp(row)

    print(
        f"verdict: {'PASS' if seed_readiness_met else 'FAIL'} "
        f"(H4 seed readiness; separability_score="
        f"{sep_stats['separability_score']})",
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
        "arm": ARM_NAME,
        "seeds": seeds,
        "training_episodes": training_episodes,
        "steps_per_episode": steps_per_episode,
        "context_switch_every": CONTEXT_SWITCH_EVERY,
        "contextmemory_gated_content_write": CONTEXTMEMORY_GATED_CONTENT_WRITE,
        "sd016_writepath_mode": SD016_WRITEPATH_MODE,
        "alpha_world": ALPHA_WORLD,
        "use_noise_floor": USE_NOISE_FLOOR,
        "num_slots": NUM_SLOTS,
        "h4_seed_readiness_floor": H4_SEED_READINESS_FLOOR,
        "h4_noise_floor": H4_NOISE_FLOOR,
        "sep_subsample_max": SEP_SUBSAMPLE_MAX,
        "probe_n": PROBE_N,
        "probe_jitter": PROBE_JITTER,
        "probe_clusters": PROBE_CLUSTERS,
    }

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        row = _run_cell(seed, full_config_slice, zg, training_episodes, steps_per_episode)
        arm_results.append(row)

    # --- P0 readiness: writepath genuinely engaged in every cell (956's own floor) ---
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
                "criteria_non_degenerate": {
                    "H4_representation_separability_measured": False,
                },
            },
            "fatal_error_count": 0,
        }
        return result, zg

    # --- H4 readiness: per-seed buffer-size floor for the separability measurement ---
    ready_rows = [row for row in arm_results if row["seed_readiness_met"]]
    n_seeds_ready = len(ready_rows)
    n_seeds_total = len(arm_results)

    if n_seeds_ready == 0:
        status = "FAIL"
        label = "h4_insufficient_write_samples"
        min_buf = min(
            min(row["n_states_safe_raw"], row["n_states_dangerous_raw"])
            for row in arm_results
        )
        result = {
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
                "preconditions": preconditions + [{
                    "name": "h4_seed_min_context_buffer_floor",
                    "measured": float(min_buf),
                    "threshold": float(H4_SEED_READINESS_FLOOR),
                    "direction": "lower",
                    "met": False,
                    "control": (
                        "not a positive-control readiness check -- this is a "
                        "sample-size floor for the pairwise-cosine estimate "
                        "itself, distinct from P0's writepath-engagement gate "
                        "(which passed above)"
                    ),
                }],
                "criteria_non_degenerate": {
                    "H4_representation_separability_measured": False,
                },
            },
            "fatal_error_count": 0,
        }
        return result, zg

    mean_separability = sum(
        row["separability_score"] for row in ready_rows
    ) / n_seeds_ready
    h4_reading = (
        "representation_separable" if mean_separability > H4_NOISE_FLOOR
        else "representation_undifferentiated"
    )

    load_bearing_passed = n_seeds_ready >= H4_N_SEEDS_REQUIRED
    criteria_non_degenerate = {
        "H4_representation_separability_measured": load_bearing_passed,
    }
    overall_pass = load_bearing_passed
    status = "PASS" if overall_pass else "FAIL"

    if overall_pass and h4_reading == "representation_separable":
        label = "h4_refuted_representation_separable"
    elif overall_pass and h4_reading == "representation_undifferentiated":
        label = "h4_supported_representation_undifferentiated"
    else:
        label = "h4_measurement_underpowered_partial_readiness"

    mean_intra_safe = sum(
        row["intra_safe_cosine"] for row in ready_rows
        if row["intra_safe_cosine"] is not None
    ) / max(sum(1 for row in ready_rows if row["intra_safe_cosine"] is not None), 1)
    mean_intra_dangerous = sum(
        row["intra_dangerous_cosine"] for row in ready_rows
        if row["intra_dangerous_cosine"] is not None
    ) / max(sum(1 for row in ready_rows if row["intra_dangerous_cosine"] is not None), 1)
    mean_inter_cosine = sum(
        row["inter_cosine"] for row in ready_rows
        if row["inter_cosine"] is not None
    ) / max(sum(1 for row in ready_rows if row["inter_cosine"] is not None), 1)

    n_seeds_tagger_moved = sum(1 for row in arm_results if row["tagger_params_moved"] is True)

    criteria = [
        {
            "name": "H4_representation_separability_measured",
            "load_bearing": True,
            "kind": "measurement",
            "passed": load_bearing_passed,
            "n_seeds_ready": n_seeds_ready,
            "n_seeds_required": H4_N_SEEDS_REQUIRED,
            "n_seeds_total": n_seeds_total,
            "note": (
                "This is a MEASUREMENT criterion, not a pass/fail on a "
                "scientific direction: it 'passes' whenever the H4 "
                "readiness precondition (per-seed buffer size "
                f">= {H4_SEED_READINESS_FLOOR}) held on >= "
                f"{H4_N_SEEDS_REQUIRED}/5 seeds, regardless of what "
                "separability_score's sign turns out to be. See "
                "interpretation.h4_reading for the separate, non-gating "
                "descriptive reading of the measured value."
            ),
        },
        {
            "name": "descriptive_mean_separability_score",
            "load_bearing": False,
            "kind": "descriptive",
            "passed": True,
            "mean_separability_score": mean_separability,
            "h4_reading": h4_reading,
            "h4_noise_floor": H4_NOISE_FLOOR,
            "n_seeds_in_mean": n_seeds_ready,
            "note": (
                "Purely descriptive -- always 'passed' (this is a "
                "measurement, not a criterion). H4_NOISE_FLOOR=0.05 is a "
                "conservative noise floor for a cosine-similarity "
                "statistic, not a rigorously derived value; stated "
                "explicitly rather than overclaiming precision."
            ),
        },
        {
            "name": "cross_cluster_cone_check",
            "load_bearing": False,
            "kind": "descriptive",
            "passed": True,
            "mean_intra_safe_cosine": mean_intra_safe,
            "mean_intra_dangerous_cosine": mean_intra_dangerous,
            "mean_inter_cosine": mean_inter_cosine,
            "note": (
                "The SD-008 '~0.98-cosine cone' reference H4's null cites: "
                "reports whether the WHOLE representation (regardless of "
                "context) sits in a tight near-1.0 cosine cone (both intra "
                "values near 1.0, would explain low separability "
                "structurally -- H4's own proposed mechanism) vs. genuinely "
                "spread out."
            ),
        },
        {
            "name": "GUMBEL_TRAINED_tagger_received_gradient",
            "load_bearing": False,
            "kind": "control",
            "passed": n_seeds_tagger_moved == len(arm_results),
            "n_seeds_moved": n_seeds_tagger_moved,
            "n_seeds_total": len(arm_results),
            "note": (
                "Mechanical positive control, copied from 956's own "
                "convention: weight=0.5 should move write_addr_tagger's "
                "parameters away from init over the real training run. "
                "Non-gating -- this experiment's own DV does not depend on "
                "whether addressing training succeeds."
            ),
        },
        {
            "name": "cross_reference_probe_2cluster_jaccard_non_gating",
            "load_bearing": False,
            "kind": "cross_reference",
            "passed": True,
            "mean_probe_2cluster_jaccard": sum(
                row["probe_2cluster_jaccard"] for row in arm_results
            ) / len(arm_results),
            "note": (
                "Non-gating context only -- 956/H1 own the load-bearing "
                "test of whether write_addr_tagger's own post-hoc, "
                "synthetic-probe content-discrimination succeeds. Reported "
                "here purely so a reader can compare against this "
                "experiment's own real-agent representation-separability "
                "reading for the same seeds."
            ),
        },
    ]

    metrics: Dict[str, Any] = {
        "n_write_calls_per_seed": [row["n_write_calls"] for row in arm_results],
        "n_states_safe_raw_per_seed": [row["n_states_safe_raw"] for row in arm_results],
        "n_states_dangerous_raw_per_seed": [row["n_states_dangerous_raw"] for row in arm_results],
        "seed_readiness_met_per_seed": [row["seed_readiness_met"] for row in arm_results],
        "intra_safe_cosine_per_seed": [row["intra_safe_cosine"] for row in arm_results],
        "intra_dangerous_cosine_per_seed": [row["intra_dangerous_cosine"] for row in arm_results],
        "inter_cosine_per_seed": [row["inter_cosine"] for row in arm_results],
        "separability_score_per_seed": [row["separability_score"] for row in arm_results],
        "probe_2cluster_jaccard_per_seed": [row["probe_2cluster_jaccard"] for row in arm_results],
        "mean_separability_score": mean_separability,
        "mean_intra_safe_cosine": mean_intra_safe,
        "mean_intra_dangerous_cosine": mean_intra_dangerous,
        "mean_inter_cosine": mean_inter_cosine,
    }

    summary_markdown = (
        f"# {QUEUE_ID} -- ContextMemory write-content H4 (input-distribution) "
        f"representation-separability probe\n\n"
        f"**Status:** {status}  **Label:** {label}\n"
        f"**Purpose:** diagnostic (claim_ids=[]; measures representation "
        f"structure at the write-stream input for hypothesis_space "
        f"contextmemory_write_content_discrimination, H4-input-distribution)\n\n"
        f"| seeds meeting H4 readiness | mean separability_score | h4_reading | "
        f"mean cross-ref Jaccard |\n"
        f"|---|---|---|---|\n"
        f"| {n_seeds_ready}/{n_seeds_total} | {mean_separability:.4f} | "
        f"{h4_reading} | "
        f"{sum(row['probe_2cluster_jaccard'] for row in arm_results) / len(arm_results):.3f} |\n\n"
        f"H4_representation_separability_measured (load-bearing, measurement "
        f"only): {'PASS' if load_bearing_passed else 'FAIL'}\n"
        f"Descriptive reading: {h4_reading} "
        f"(mean_separability_score={mean_separability:.4f} vs noise floor "
        f"{H4_NOISE_FLOOR})\n"
    )

    result = {
        "outcome": status,
        "status": status,
        "claim_ids": CLAIM_IDS,
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "non_contributory",
        "sleep_driver_pattern": "N/A (no sleep loop)",
        "metrics": metrics,
        "arm_results": arm_results,
        "summary_markdown": summary_markdown,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "h4_reading": h4_reading,
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
        "arm": ARM_NAME,
        "write_selection": WRITE_SELECTION,
        "contextmemory_write_addressing_loss_weight": WRITE_ADDRESSING_LOSS_WEIGHT,
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
        "h4_seed_readiness_floor": H4_SEED_READINESS_FLOOR,
        "h4_n_seeds_required": H4_N_SEEDS_REQUIRED,
        "h4_noise_floor": H4_NOISE_FLOOR,
        "sep_subsample_max": SEP_SUBSAMPLE_MAX,
        "max_state_buf": MAX_STATE_BUF,
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
