#!/opt/local/bin/python3
"""
V3-EXQ-888 -- MECH-074 PARENT read/write-head route dissociation (wall-independent).

Confirming DV for the PARENT claim MECH-074 ("Amygdala functions as read/write head
for valenced hippocampal map"), routed from IGW-20260803-230 / backlog EVB-0074.
GOV-CONFIRM-1: MECH-074 has a built substrate (SD-035 BLAAnalog + CeAAnalog,
IMPLEMENTED 2026-04-21), lit_confidence 0.865, and ZERO experimental entries of any
kind at the PARENT granularity -- every recorded MECH-074* run tags a CHILD only.

SCOPE DECISION (stated explicitly per the parent/child tagging rule). The parent's
own notes say governance should primarily evaluate the children and that the parent
"remains a cross-cutting umbrella". So this probe does NOT re-test a child. It tests
the one thing that is the PARENT's own content and that no child asserts alone:
that the amygdala operates as a HEAD over a SHARED store -- it writes on TWO
functionally separable routes into the SAME hippocampal exploration buffer, and its
READ is addressed by what its WRITE deposited there. A single-route result would
evidence a child, not the parent; the parent needs the SEPARABILITY.

  WRITE-STRENGTH route (MECH-074a): BLAAnalog.encoding_gain -> Trajectory.memory_strength
  WRITE-ADDRESS route (MECH-074b) : BLAAnalog.arousal_tag -> Trajectory.arousal_tag
                                     -> get_exploration_arousal_tags()
                                     -> BLAAnalog.retrieval_bias (w_i = 1 + alpha*tag_i)
  SHARED READ                      : HippocampalModule._sample_exploration_trajectory,
                                     weights = memory_strength * retrieval_bias
                                     (module.py ~2462-2485; index-aligned to the buffer)

SUBSTRATE WIRING VERIFIED (/queue-experiment Step 2.5 + 2.5a probe, 2026-08-03):
  agent.py ~4404  arousal_tags_in_context = hippocampal.get_exploration_arousal_tags()
  agent.py ~4434  bla.tick(..., arousal_tags_in_context=...) -> retrieval_bias
  agent.py ~8945/10680  diverse_replay(..., retrieval_bias=...) -> the weighted sample
  agent.py ~3380  memory_strength = episode peak encoding_gain; arousal_tag = peak tag
  Direct BLAAnalog probe confirmed both ablation knobs behave as documented:
    bla_encoding_gain_max=1.0            -> encoding_gain == 1.0 at every arousal
    bla_retrieval_tag_at_encoding=False  -> arousal_tag == 0.0 at every arousal
    retrieval_bias is a PER-TRACE vector 1 + 0.6*tags (not a scalar), as MECH-074b
    requires, and is computed from the tags supplied in context.

WHY THIS IS NOT V3-EXQ-659 AGAIN. 659 (MECH-074a, PASS supports) measured exactly ONE
route and said so: "retrieval_bias is deliberately NOT supplied at readout so the only
weight driving the sample is memory_strength -- the clean 074a isolation". This probe
supplies retrieval_bias at readout (the wired read path 659 excluded) and adds the two
arms 659 does not have -- GAIN_FLAT and TAG_OFF -- which is what turns a single-route
measurement into a route dissociation.

WHY THE DV IS NOT THE SUBSTRATE'S OWN TAG (the load-bearing instrument decision).
659's AOR was measured against Trajectory.arousal_tag. That statistic is IDENTICALLY
ZERO in any tag-ablated arm, so it cannot compare arms across the address ablation --
it would score TAG_OFF as vacuous rather than as a route with the address channel
removed. This probe therefore records its OWN GROUND-TRUTH arousal label per trace,
in the harness, independent of the substrate: gt_arousal = the peak ||z_harm_a||
observed during that episode. It is defined identically in all four arms, so all four
AORs are on one footing.

DESIGN -- 4 arms x 3 distinct seeds [42, 43, 45]. (Seed 44 is excluded per the standing
per-seed instability finding on this reef-config env family; 45 substituted.)
  ARM_FULL      gain_max=2.5, tag ON   -> both routes live
  ARM_TAG_OFF   gain_max=2.5, tag OFF  -> write-STRENGTH route only (address severed)
  ARM_GAIN_FLAT gain_max=1.0, tag ON   -> write-ADDRESS route only (strength pinned)
  ARM_OFF       gain_max=1.0, tag OFF  -> neither route (negative control)
Everything else is identical across arms. CeA is off (not under test).

Per cell: (1) clear the exploration buffer; (2) COLLECT episodes ALTERNATING threat /
neutral context, one exploration trace per episode, harness recording that episode's
peak ||z_harm_a|| as gt_arousal aligned to the buffer position; (3) READOUT: build
retrieval_bias through the REAL substrate path (bla.tick with the live buffer tags),
then draw READOUT_SAMPLES times from _sample_exploration_trajectory(retrieval_bias=...)
and measure AOR = mean(gt_arousal | sampled) - mean(gt_arousal | buffer).

NO WARMUP / UNTRAINED ENCODER -- inherited deliberately from 659, same reasoning: the
untrained AffectiveHarmEncoder maps harm magnitude monotonically to ||z_harm_a|| with a
RESTING norm (~0.31) below the BLA threshold (0.4), while training the harm-accum aux
loss inflates the resting norm to ~1.2 and saturates the threshold in EVERY context,
destroying the threat/neutral arousal contrast the DV needs. The mechanism under test
is BLA arithmetic over a controlled arousal contrast, not learned encoder semantics.

AOR is STANDARDISED by the buffer's own gt_arousal std before the criteria are applied
(AOR_z), because the absolute scale of ||z_harm_a|| is an arbitrary property of the
untrained encoder and an absolute margin would silently encode an assumption about it.
The divisor is a property of the label distribution, identical across arms by
construction -- the four arms share one collection and differ only in readout weighting.

PRE-REGISTERED PREDICTIONS (per seed; cohort needs SEEDS_PASS_MIN of 3):
  C1 head-level authority        AOR_z_FULL      - AOR_z_OFF > AOR_Z_MARGIN
  C2 address route has authority AOR_z_GAIN_FLAT - AOR_z_OFF > AOR_Z_MARGIN [MECH-074b]
  C3 strength route has authority AOR_z_TAG_OFF  - AOR_z_OFF > AOR_Z_MARGIN [MECH-074a]
  C4 SEPARABILITY (load-bearing for the PARENT) = C2 AND C3: BOTH routes carry
     independent authority over the same store, i.e. neither is wholly redundant.
  PASS iff C1 AND C4.

C2 is the genuinely uncertain quantity and is NOT fixed by construction: with
encoding_gain pinned at 1.0 the address route must survive a full round trip through
the shared store -- tag written at encode, stored on the trajectory, read back by
get_exploration_arousal_tags in buffer order, turned into a per-trace bias by BLA,
and consumed index-aligned at sampling. Any staleness or misalignment anywhere in that
loop yields AOR ~ 0 with everything still "wired". C4 can also fail in the informative
direction: if AOR_TAG_OFF ~ AOR_FULL, the address route adds nothing on top of gain and
the READ half of the read/write head is not supported.

DV-SYMMETRY DECLARATION (mandatory; one line per arm). The DV is a mean of a continuous
harness label over a multinomial-weighted sample of traces. Its symmetry group is
(i) UNIFORM POSITIVE RESCALING of the whole weight vector -- the multinomial normalises,
so a manipulation that scaled every trace's weight equally would be invisible -- and
(ii) PERMUTATION of traces, under which the DV is NOT invariant because gt_arousal is
attached to trace identity rather than pooled.
  ARM_FULL vs ARM_OFF      : not invariant. Both routes move weight DIFFERENTIALLY --
                             only above-threshold episodes gain elevated encoding_gain
                             and a non-zero tag; low-arousal traces are untouched.
  ARM_GAIN_FLAT vs ARM_OFF : not invariant. tag ON makes retrieval_bias = 1 + 0.6*tag_i,
                             a per-trace NON-UNIFORM vector (probe-confirmed
                             [1.0, 1.3, 1.6, 1.15]); tag OFF collapses it to all-ones.
                             Removing non-uniformity is not a uniform rescale.
  ARM_TAG_OFF vs ARM_OFF   : not invariant. gain_max 2.5 -> 1.0 changes memory_strength
                             only on arousal-crossing episodes, again differentially.
  ARM_OFF                  : the negative control; both routes uniform by construction,
                             which is the prediction (AOR ~ 0), not a hidden invariance.
No arm's manipulation is invariant under the DV's symmetry group.

GOV-REUSE-1 (Step 2.4). Decisive readout: per-arm AOR against a harness ground-truth
arousal label under a gain-flat and a tag-off ablation. Checked
v3_exq_659_mech074a_bla_encoding_gain_replay_bias_20260609T200751Z_v3 (the only prior
run carrying an AOR readout) and v3_exq_473/474_sd035_* : 659 has only PRIMARY/ABLATION
(no gain-flat, no tag-off arm), measures AOR against the substrate tag rather than a
ground-truth label, and omits retrieval_bias at readout entirely; it also records no
top-level substrate_hash (pre-standard), so substrate compatibility is UNVERIFIABLE and
is treated as not-recoverable. 473/474 are diagnostic probes (scoring_excluded) with no
AOR readout. Not recoverable -> run.

experiment_purpose: evidence. claim_ids: MECH-074 (parent, on C4), MECH-074a (C3),
MECH-074b (C2) -- with evidence_direction_per_claim, since the three criteria can
resolve differently. The parent is tagged ONLY because C4 tests the parent's own
separability content; a single-route outcome routes the parent to mixed/weakens, never
to supports.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from _harness import StepHarness  # noqa: E402
from _lib.arm_fingerprint import arm_cell  # noqa: E402
from _lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from _lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_888_mech074_readwrite_head_route_dissociation"
QUEUE_ID = "V3-EXQ-888"
CLAIM_IDS: List[str] = ["MECH-074", "MECH-074a", "MECH-074b"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 excluded: recurring per-seed early-episode-death instability on this
# reef-config env family (EXQ-539-540, V3-EXQ-538a). 45 substituted.
SEEDS = [42, 43, 45]

# Encoder dims (matched to V3-EXQ-659 so the arousal calibration carries over).
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32       # z_harm (SD-010 sensory)
HARM_A_DIM = 16     # z_harm_a (SD-011 affective)
HARM_HISTORY_LEN = 10

COLLECT_EPISODES = 40   # one exploration trace per episode; buffer maxlen is 50
STEPS_PER_EPISODE = 60
# Multinomial draws over the exploration buffer. Large on purpose and essentially
# free: the buffer and the weight vector are FIXED by the time the readout runs, so
# more draws only estimate the deterministic weighted mean more precisely -- they
# cannot inflate an effect. At N draws the standard error of AOR is ~sigma/sqrt(N),
# i.e. ~0.007 sigma here, which is what makes the 0.03-sigma margin below a ~4-SE gate
# rather than the ~2-SE one a smaller N would give.
READOUT_SAMPLES = 20000
TOTAL_EPISODES = COLLECT_EPISODES  # = episodes_per_run

# BLA calibration (identical across arms; only the two ablation knobs differ).
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5     # short post-event window -> per-episode gain reflects THAT
                         # episode's arousal (bla.reset() also called per collect ep).
BLA_RETRIEVAL_BIAS_ALPHA = 0.6

# ---- Pre-registered acceptance thresholds (constants, never derived post-hoc) ----
MIN_TRACES = 8            # readiness: buffer must hold enough traces
GT_AROUSAL_STD_FLOOR = 0.02   # readiness: the DV's own label must VARY across traces
MS_RANGE_FLOOR = 0.05     # readiness (gain-live arms): strength route non-uniform
RB_RANGE_FLOOR = 0.01     # readiness (tag-live arms): address route non-uniform
# The criteria are expressed in STANDARDISED units -- AOR divided by the buffer's own
# gt_arousal std -- NOT in raw ||z_harm_a|| units. Rationale, and it is a correction
# made before the run rather than a threshold tuned to it: gt_arousal is a latent norm
# whose absolute scale is an arbitrary property of the untrained encoder (measured
# std ~0.05 in the smoke), so any absolute margin silently encodes an assumption about
# that scale. Standardising makes the criterion an effect size, so the threshold means
# the same thing whatever the encoder's scale happens to be. The DIVISOR is a property
# of the buffer (the label distribution), never of the arms or of the outcome, and it
# is identical across arms by construction -- the four arms differ only in the readout
# weighting, so they share one collection and one label distribution.
AOR_Z_MARGIN = 0.03       # per-seed contrast floor vs ARM_OFF, in gt_arousal std units
OFF_AOR_Z_FLAT_MAX = 0.02  # sanity: |AOR_OFF| stays near zero, same units
SEEDS_PASS_MIN = 2        # >= 2/3 seeds

ARM_FULL = "ARM_FULL"
ARM_TAG_OFF = "ARM_TAG_OFF"
ARM_GAIN_FLAT = "ARM_GAIN_FLAT"
ARM_OFF = "ARM_OFF"

ARMS: List[Dict[str, Any]] = [
    {"arm_id": ARM_FULL, "gain_max": 2.5, "tag_on": True},
    {"arm_id": ARM_TAG_OFF, "gain_max": 2.5, "tag_on": False},
    {"arm_id": ARM_GAIN_FLAT, "gain_max": 1.0, "tag_on": True},
    {"arm_id": ARM_OFF, "gain_max": 1.0, "tag_on": False},
]

# Threat context: SD-022 scheduled limb-damage injection drives RELIABLE,
# policy-independent body damage -> harm_obs_a rises -> ||z_harm_a|| crosses the BLA
# arousal threshold. Scheduled injection (not just hazard contact) is used because the
# agent is UNTRAINED and a frozen policy does not reliably accumulate enough contact
# damage to clear the threshold.
THREAT_ENV_KWARGS: Dict[str, Any] = dict(
    size=10,
    num_hazards=4,
    num_resources=4,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    use_proxy_fields=True,
    toroidal=False,
    harm_history_len=HARM_HISTORY_LEN,
    limb_damage_enabled=True,
    damage_increment=0.15,
    failure_prob_scale=0.3,
    heal_rate=0.002,
    scheduled_limb_damage_enabled=True,
    scheduled_limb_damage_interval=8,
    scheduled_limb_damage_prob=1.0,
    scheduled_limb_damage_magnitude=0.25,
    scheduled_limb_damage_limb_selection="all",
)
# Neutral context: no hazards AND no scheduled injection -> low ||z_harm_a|| (~0.3,
# below the 0.4 threshold). Same obs dims as threat so one agent runs in both. The
# threat/neutral MIX is what produces the cross-trace gt_arousal variance the DV needs.
NEUTRAL_ENV_KWARGS: Dict[str, Any] = dict(
    THREAT_ENV_KWARGS, num_hazards=0, scheduled_limb_damage_enabled=False,
)


# --------------------------------------------------------------------------------
# Readiness preconditions. Regime-conditioned with applies_to: MS_RANGE is structurally
# impossible in the gain-pinned arms and RB_RANGE is structurally impossible in the
# tag-ablated arms, and ANDing those whole-run would vacate the very arms this design
# exists to compare (the V3-EXQ-785 defect). They are scoped OUT of the arms where the
# quantity is not meaningful, NOT failed by them.
#
# Each precondition asserts the SAME STATISTIC its dependent criterion routes on:
# the criteria are differences of MEANS of gt_arousal, so the readiness check is the
# SPREAD of gt_arousal (a zero-spread label makes every AOR identically 0 regardless of
# any weighting); the route non-uniformity checks are RANGES because a route with zero
# range across traces cannot move a weighted sample at all.
# --------------------------------------------------------------------------------
PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="n_traces_sufficient",
        description="Exploration buffer holds enough traces for a weighted sample.",
        control="collect phase alternating threat/neutral, one trace per episode",
        threshold=float(MIN_TRACES - 1),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="gt_arousal_spread",
        description=(
            "Std of the harness ground-truth arousal label across buffer traces. This "
            "is the SAME statistic every criterion routes on: all four AORs are "
            "differences of means of gt_arousal, so a zero-spread label forces every "
            "AOR to 0 no matter how the sample is weighted."
        ),
        control="threat/neutral alternation gives high- and low-arousal episodes",
        threshold=GT_AROUSAL_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="buffer_gt_alignment",
        description=(
            "Harness gt_arousal list length equals the buffer length, so label index i "
            "really is trace i. Misalignment would silently scramble the DV."
        ),
        control="one flush per episode, verified by buffer growth at each agent.reset()",
        threshold=0.5,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="strength_route_nonuniform",
        description=(
            "memory_strength range across buffer traces. The write-STRENGTH route can "
            "only move a weighted sample if encoding_gain actually varies trace to "
            "trace."
        ),
        control="threat episodes cross the BLA arousal threshold and elevate the gain",
        threshold=MS_RANGE_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["gain_max"] > 1.0),
        applies_note=(
            "not meaningful with bla_encoding_gain_max=1.0: the strength route is "
            "pinned uniform BY DESIGN in this arm, which is the manipulation, not a "
            "substrate failure"
        ),
        structural_max=lambda ctx: (0.0 if ctx["gain_max"] <= 1.0 else None),
    ),
    PreconditionSpec(
        name="address_route_nonuniform",
        description=(
            "retrieval_bias range across buffer traces, taken from the REAL BLA output. "
            "The write-ADDRESS route can only move a weighted sample if the per-trace "
            "bias vector actually varies."
        ),
        control="threat episodes write a non-zero arousal_tag; neutral episodes write 0",
        threshold=RB_RANGE_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["tag_on"]),
        applies_note=(
            "not meaningful with bla_retrieval_tag_at_encoding=False: every tag is 0 "
            "and the bias is all-ones BY DESIGN in this arm, which is the manipulation"
        ),
        structural_max=lambda ctx: (0.0 if not ctx["tag_on"] else None),
    ),
]


# z_goal liveness recording. A fresh agent is built inside each cell, so the
# accumulator is used rather than holding every arm x seed agent alive to the end.
_ZG = ZGoalStreamAccumulator()


def _arm_ctx(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "gain_max": float(arm["gain_max"]),
        "tag_on": bool(arm["tag_on"]),
    }


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2, gain_max: float, tag_on: bool) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=WORLD_DIM,
        self_dim=SELF_DIM,
        use_harm_stream=True,
        z_harm_dim=HARM_DIM,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        limb_damage_enabled=True,
        damage_increment=float(THREAT_ENV_KWARGS["damage_increment"]),
        failure_prob_scale=float(THREAT_ENV_KWARGS["failure_prob_scale"]),
        heal_rate=float(THREAT_ENV_KWARGS["heal_rate"]),
        # SD-035 amygdala analogue -- BLA on; CeA off (not under test).
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=False,
        bla_encoding_gain_max=float(gain_max),
        bla_encoding_gain_floor=1.0,
        bla_arousal_threshold_on=BLA_AROUSAL_THRESHOLD_ON,
        bla_arousal_peak=BLA_AROUSAL_PEAK,
        bla_window_steps=BLA_WINDOW_STEPS,
        bla_retrieval_bias_alpha=BLA_RETRIEVAL_BIAS_ALPHA,
        bla_retrieval_tag_at_encoding=bool(tag_on),
        # MECH-165 / SD-035 read/write head: exploration-trace recording on so
        # encoding_gain lands as memory_strength on each flushed trajectory.
        replay_diversity_enabled=True,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _collect_traces(
    agent: REEAgent,
    threat_env: CausalGridWorldV2,
    neutral_env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    label: str,
) -> Dict[str, Any]:
    """Run frozen-encoder episodes ALTERNATING threat/neutral, recording one exploration
    trace per episode AND the harness ground-truth arousal label for that trace.

    gt_arousal is the peak ||z_harm_a|| observed during the episode, measured by this
    harness and NOT read from the substrate -- so it is defined identically in the
    tag-ablated arms, where Trajectory.arousal_tag is identically 0.

    Alignment: agent.reset() flushes the PREVIOUS episode's trace into the buffer, so
    the pending label is appended at the moment the buffer is observed to grow. Buffer
    growth is asserted to be exactly one trace per flush; a final reset flushes the last
    episode. FIFO eviction (buffer maxlen 50 vs 40 episodes here) would drop from the
    FRONT, so the label list is trimmed from the front to match if it ever fires.
    """
    buf = agent.hippocampal._exploration_buffer
    buf.clear()
    agent.eval()

    gt_by_index: List[float] = []
    pending_gt: Optional[float] = None
    multi_flush_events = 0

    max_threat_zharm_a = 0.0
    max_neutral_zharm_a = 0.0
    n_threat = 0
    n_neutral = 0
    last_z_harm_a: Optional[torch.Tensor] = None

    for ep in range(num_episodes):
        is_threat = (ep % 2 == 0)
        env = threat_env if is_threat else neutral_env
        harness = StepHarness(agent, env, train_mode=False, seed=seed + ep)
        _, obs_dict = env.reset()

        n_before = len(buf)
        agent.reset()   # flushes the PREVIOUS episode's exploration trace
        grew = len(buf) - n_before
        if grew > 1:
            multi_flush_events += 1
        if grew >= 1 and pending_gt is not None:
            gt_by_index.append(float(pending_gt))
            pending_gt = None

        if agent.bla is not None:
            agent.bla.reset()
        harness.reset()

        ep_peak = 0.0
        for _ in range(steps_per_episode):
            result = harness.step(obs_dict)
            lat = result.latent
            if lat is not None and getattr(lat, "z_harm_a", None) is not None:
                zt = lat.z_harm_a.detach()
                last_z_harm_a = zt
                zn = float(torch.linalg.norm(zt.flatten()).item())
                ep_peak = max(ep_peak, zn)
                if is_threat:
                    max_threat_zharm_a = max(max_threat_zharm_a, zn)
                else:
                    max_neutral_zharm_a = max(max_neutral_zharm_a, zn)
            obs_dict = result.next_obs_dict
            if result.done:
                break

        pending_gt = ep_peak
        if is_threat:
            n_threat += 1
        else:
            n_neutral += 1
        if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
            print(f"  [train] {label} ep {ep + 1}/{num_episodes}", flush=True)

    # Final flush of the last episode's trajectory.
    n_before = len(buf)
    agent.reset()
    grew = len(buf) - n_before
    if grew > 1:
        multi_flush_events += 1
    if grew >= 1 and pending_gt is not None:
        gt_by_index.append(float(pending_gt))
        pending_gt = None

    # FIFO eviction drops from the front; mirror that on the label list.
    if len(gt_by_index) > len(buf):
        gt_by_index = gt_by_index[len(gt_by_index) - len(buf):]

    n_gain_elev = int(getattr(agent.bla, "_n_gain_elevations", 0)) if agent.bla else 0
    return {
        "gt_by_index": gt_by_index,
        "last_z_harm_a": last_z_harm_a,
        "max_threat_zharm_a_norm": float(max_threat_zharm_a),
        "max_neutral_zharm_a_norm": float(max_neutral_zharm_a),
        "n_threat_episodes": int(n_threat),
        "n_neutral_episodes": int(n_neutral),
        "n_gain_elevations": n_gain_elev,
        "multi_flush_events": int(multi_flush_events),
    }


def _readout(
    agent: REEAgent,
    coll: Dict[str, Any],
    *,
    seed: int,
    n_samples: int,
) -> Dict[str, Any]:
    """Draw from the SHARED read path and measure arousal over-representation.

    retrieval_bias is produced by the REAL substrate path -- BLAAnalog.tick fed with
    the live buffer tags from get_exploration_arousal_tags() -- rather than recomputed
    in the harness, so the round trip through the shared store is what is being tested.
    """
    buf = agent.hippocampal._exploration_buffer
    n_traces = len(buf)
    gt = [float(x) for x in coll["gt_by_index"]]

    buffer_ms = [float(getattr(t, "memory_strength", 1.0)) for t in buf]
    buffer_tag = [float(getattr(t, "arousal_tag", 0.0)) for t in buf]

    result: Dict[str, Any] = {
        "n_traces": int(n_traces),
        "n_gt_labels": len(gt),
        "buffer_gt_alignment_ok": bool(n_traces > 0 and len(gt) == n_traces),
        "buffer_mean_gt_arousal": float(np.mean(gt)) if gt else 0.0,
        "buffer_std_gt_arousal": float(np.std(gt)) if gt else 0.0,
        "buffer_mean_tag": float(np.mean(buffer_tag)) if n_traces else 0.0,
        "memory_strength_min": float(min(buffer_ms)) if n_traces else 1.0,
        "memory_strength_max": float(max(buffer_ms)) if n_traces else 1.0,
        "memory_strength_range": (
            float(max(buffer_ms) - min(buffer_ms)) if n_traces else 0.0),
        "retrieval_bias_min": 1.0,
        "retrieval_bias_max": 1.0,
        "retrieval_bias_range": 0.0,
        "retrieval_bias_supplied": False,
        "sampled_mean_gt_arousal": 0.0,
        "arousal_over_representation": 0.0,
        "arousal_over_representation_z": 0.0,
        "n_sampled": 0,
        # Secondary, coarser DV: threat-context over-representation. Recorded
        # generously; the pre-registered criteria route on the continuous DV.
        "buffer_threat_frac": 0.0,
        "sampled_threat_frac": 0.0,
        "threat_over_representation": 0.0,
    }
    if n_traces < 2 or len(gt) != n_traces:
        return result

    # --- Build retrieval_bias through the real substrate path. ---
    retrieval_bias = None
    tags = agent.hippocampal.get_exploration_arousal_tags()
    z_last = coll.get("last_z_harm_a")
    if agent.bla is not None and tags is not None and z_last is not None:
        out = agent.bla.tick(
            z_harm_a=z_last,
            arousal_tags_in_context=tags,
        )
        retrieval_bias = out.retrieval_bias
    if retrieval_bias is not None and int(retrieval_bias.numel()) == n_traces:
        rb = [float(x) for x in retrieval_bias.detach().flatten().tolist()]
        result["retrieval_bias_supplied"] = True
        result["retrieval_bias_min"] = float(min(rb))
        result["retrieval_bias_max"] = float(max(rb))
        result["retrieval_bias_range"] = float(max(rb) - min(rb))
    else:
        retrieval_bias = None

    # Threat/neutral base rate: episodes alternate starting with threat, and the
    # buffer holds them in episode order.
    threat_flags = [1.0 if (i % 2 == 0) else 0.0 for i in range(n_traces)]
    result["buffer_threat_frac"] = float(np.mean(threat_flags))

    # Deterministic readout RNG (seed-derived) for reproducibility.
    torch.manual_seed(seed + 100000)
    sampled_gt: List[float] = []
    sampled_threat: List[float] = []
    # Identity map from trace object to buffer index, so a drawn trajectory can be
    # scored against its harness label. Buffer entries are distinct objects.
    index_of = {id(t): i for i, t in enumerate(buf)}
    for _ in range(n_samples):
        traj = agent.hippocampal._sample_exploration_trajectory(
            retrieval_bias=retrieval_bias
        )
        if traj is None:
            continue
        idx = index_of.get(id(traj))
        if idx is None:
            continue
        sampled_gt.append(gt[idx])
        sampled_threat.append(threat_flags[idx])

    if sampled_gt:
        s_mean = float(np.mean(sampled_gt))
        result["n_sampled"] = len(sampled_gt)
        result["sampled_mean_gt_arousal"] = s_mean
        aor = s_mean - result["buffer_mean_gt_arousal"]
        result["arousal_over_representation"] = aor
        sd = float(result["buffer_std_gt_arousal"])
        result["arousal_over_representation_z"] = (aor / sd) if sd > 1e-9 else 0.0
        st = float(np.mean(sampled_threat))
        result["sampled_threat_frac"] = st
        result["threat_over_representation"] = st - result["buffer_threat_frac"]
    return result


def _run_cell(seed: int, arm: Dict[str, Any], *, dry: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    collect_eps = 8 if dry else COLLECT_EPISODES
    steps = 20 if dry else STEPS_PER_EPISODE
    n_samples = 400 if dry else READOUT_SAMPLES

    config_slice = {
        "arm_id": arm_id,
        "bla_encoding_gain_max": float(arm["gain_max"]),
        "bla_encoding_gain_floor": 1.0,
        "bla_retrieval_tag_at_encoding": bool(arm["tag_on"]),
        "bla_retrieval_bias_alpha": BLA_RETRIEVAL_BIAS_ALPHA,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "world_dim": WORLD_DIM,
        "harm_a_dim": HARM_A_DIM,
        "collect_eps": collect_eps,
        "steps_per_episode": steps,
        "readout_samples": n_samples,
    }
    label = f"mech074 seed={seed} arm={arm_id}"
    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        # No extra ineligibility reason: each cell builds its OWN env and agent and
        # arm_cell resets all RNG on entry, so a cell IS a pure function of
        # (substrate, config_slice, seed) -- nothing mutable is shared across cells.
        # (V3-EXQ-659 declared "stateful_replay_buffer_readout" here; that does not
        # apply to this driver, where the buffer lives inside the per-cell agent and
        # the readout only READS it.) The fingerprint is driver-INCLUSIVE (the
        # default), so a same-driver successor can reuse these cells; cross-driver
        # reuse would additionally need the OFF path factored into
        # experiments/_lib/baselines/, which is NOT done here and is recorded as
        # follow-on rather than claimed.
    ) as cell:
        threat_env = _make_env(seed, THREAT_ENV_KWARGS)
        neutral_env = _make_env(seed + 1, NEUTRAL_ENV_KWARGS)
        cfg = _build_config(threat_env, float(arm["gain_max"]), bool(arm["tag_on"]))
        agent = REEAgent(cfg)

        coll = _collect_traces(
            agent, threat_env, neutral_env,
            num_episodes=collect_eps, steps_per_episode=steps,
            seed=seed, label=label,
        )
        read = _readout(agent, coll, seed=seed, n_samples=n_samples)
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time

        coll_public = {k: v for k, v in coll.items()
                       if k not in ("gt_by_index", "last_z_harm_a")}
        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            "gain_max": float(arm["gain_max"]),
            "tag_on": bool(arm["tag_on"]),
            **coll_public,
            **read,
            # Generous recording: the full per-trace label vector, so a successor can
            # re-derive any other over-representation statistic without re-running.
            "gt_arousal_by_trace": [float(x) for x in coll["gt_by_index"]],
        }
        cell.stamp(row)

    gate = evaluate_arm_gate(
        arm_id,
        _arm_ctx(arm),
        PRECONDITION_SPECS,
        measured={
            "n_traces_sufficient": float(row["n_traces"]),
            "gt_arousal_spread": float(row["buffer_std_gt_arousal"]),
            "buffer_gt_alignment": 1.0 if row["buffer_gt_alignment_ok"] else 0.0,
            "strength_route_nonuniform": float(row["memory_strength_range"]),
            "address_route_nonuniform": float(row["retrieval_bias_range"]),
        },
    )
    row["arm_gate"] = gate

    # Per-cell verdict (progress only; the cohort evaluation is authoritative).
    aor = float(row["arousal_over_representation_z"])
    if arm_id == ARM_OFF:
        cell_pass = bool(gate["gate_green"] and abs(aor) <= OFF_AOR_Z_FLAT_MAX)
    else:
        cell_pass = bool(gate["gate_green"] and aor > 0.0)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    row["cell_pass"] = bool(cell_pass)
    return row


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = r

    per_seed: List[Dict[str, Any]] = []
    c1_ok = c2_ok = c3_ok = 0
    off_flat_ok = 0
    for seed, arms in sorted(by_seed.items()):
        full = arms.get(ARM_FULL)
        tag_off = arms.get(ARM_TAG_OFF)
        gain_flat = arms.get(ARM_GAIN_FLAT)
        off = arms.get(ARM_OFF)
        if full is None or tag_off is None or gain_flat is None or off is None:
            continue
        aor_full = float(full["arousal_over_representation_z"])
        aor_tag_off = float(tag_off["arousal_over_representation_z"])
        aor_gain_flat = float(gain_flat["arousal_over_representation_z"])
        aor_off = float(off["arousal_over_representation_z"])

        d_full = aor_full - aor_off
        d_gain_flat = aor_gain_flat - aor_off
        d_tag_off = aor_tag_off - aor_off

        s_c1 = bool(d_full > AOR_Z_MARGIN)
        s_c2 = bool(d_gain_flat > AOR_Z_MARGIN)
        s_c3 = bool(d_tag_off > AOR_Z_MARGIN)
        s_flat = bool(abs(aor_off) <= OFF_AOR_Z_FLAT_MAX)
        c1_ok += int(s_c1)
        c2_ok += int(s_c2)
        c3_ok += int(s_c3)
        off_flat_ok += int(s_flat)

        per_seed.append({
            "seed": seed,
            # z-scored (the criteria route on these)
            "aor_z_full": aor_full,
            "aor_z_tag_off": aor_tag_off,
            "aor_z_gain_flat": aor_gain_flat,
            "aor_z_off": aor_off,
            "delta_full_vs_off": d_full,
            "delta_gain_flat_vs_off": d_gain_flat,
            "delta_tag_off_vs_off": d_tag_off,
            # raw gt_arousal units, retained so the standardisation is invertible
            "aor_raw_full": float(full["arousal_over_representation"]),
            "aor_raw_tag_off": float(tag_off["arousal_over_representation"]),
            "aor_raw_gain_flat": float(gain_flat["arousal_over_representation"]),
            "aor_raw_off": float(off["arousal_over_representation"]),
            "buffer_std_gt_arousal": float(full["buffer_std_gt_arousal"]),
            "threat_or_full": float(full["threat_over_representation"]),
            "threat_or_gain_flat": float(gain_flat["threat_over_representation"]),
            "threat_or_tag_off": float(tag_off["threat_over_representation"]),
            "threat_or_off": float(off["threat_over_representation"]),
            "c1_head_authority": s_c1,
            "c2_address_route_authority": s_c2,
            "c3_strength_route_authority": s_c3,
            "off_flat_ok": s_flat,
            "seed_pass": bool(s_c1 and s_c2 and s_c3),
        })

    n_seeds = len(per_seed)
    # Well-defined for any seed count: with the pre-registered 3-seed cohort this is
    # exactly SEEDS_PASS_MIN (2 of 3). It only binds differently when fewer seeds ran
    # than were pre-registered (a smoke), where an unreachable criterion would
    # otherwise force a misleading "no authority" label on a run that never had the
    # seeds to test it.
    seeds_needed = min(SEEDS_PASS_MIN, n_seeds) if n_seeds else SEEDS_PASS_MIN
    c1_met = c1_ok >= seeds_needed
    c2_met = c2_ok >= seeds_needed
    c3_met = c3_ok >= seeds_needed
    c4_met = bool(c2_met and c3_met)   # SEPARABILITY -- the parent's own content
    outcome_pass = bool(c1_met and c4_met)

    # Per-claim directions. The parent is supported ONLY by C4 (separability); a
    # single-route result evidences a child, never the parent.
    if c4_met and c1_met:
        parent_dir = "supports"
    elif c1_met and (c2_met or c3_met):
        # The head moves replay, but one route carries all of it -> the two-route
        # read/write-head picture is not what the substrate shows.
        parent_dir = "weakens"
    else:
        parent_dir = "mixed"

    per_claim = {
        "MECH-074": parent_dir,
        "MECH-074a": "supports" if c3_met else "weakens",
        "MECH-074b": "supports" if c2_met else "weakens",
    }
    if outcome_pass:
        overall_dir = "supports"
    elif parent_dir == "weakens" or not (c2_met or c3_met):
        overall_dir = "mixed"
    else:
        overall_dir = "mixed"

    if outcome_pass:
        label = "mech074_readwrite_head_two_route_separability_supports"
    elif c1_met and c3_met and not c2_met:
        label = "mech074_strength_route_only_address_route_no_authority"
    elif c1_met and c2_met and not c3_met:
        label = "mech074_address_route_only_strength_route_no_authority"
    elif not c1_met:
        label = "mech074_head_no_replay_authority"
    else:
        label = "mech074_route_dissociation_inconclusive"

    return {
        "outcome": "PASS" if outcome_pass else "FAIL",
        "evidence_direction": overall_dir,
        "evidence_direction_per_claim": per_claim,
        "interpretation_label": label,
        "n_seeds": n_seeds,
        "seeds_needed": int(seeds_needed),
        "c1_head_authority_met": c1_met,
        "c2_address_route_authority_met": c2_met,
        "c3_strength_route_authority_met": c3_met,
        "c4_separability_met": c4_met,
        "c1_seeds_ok": c1_ok,
        "c2_seeds_ok": c2_ok,
        "c3_seeds_ok": c3_ok,
        "off_flat_seeds_ok": off_flat_ok,
        "combination_rule": (
            "PASS iff C1 AND C4, where C4 = (C2 AND C3). Each criterion is met when it "
            "holds on >= SEEDS_PASS_MIN of the seeds. C4 (separability) is the "
            "load-bearing criterion for the PARENT claim MECH-074; C3 alone is "
            "MECH-074a and C2 alone is MECH-074b."
        ),
        "per_seed": per_seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    seeds = SEEDS[:1] if args.dry_run else SEEDS

    # Design-time refusal: prove no arm faces a precondition it cannot satisfy in a
    # way that would silently vacate the run (the V3-EXQ-785 check, run BEFORE compute).
    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, [_arm_ctx(a) for a in ARMS]
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(seed, arm, dry=args.dry_run))

    ev = _evaluate(rows)

    # ---- Engagement assertions. Under --dry-run these are the cheap proof that the
    # decisive readout is non-trivially exercised BEFORE the full seed x arm grid is
    # committed: the label must vary, each route must be non-uniform in exactly the
    # arms that carry it (and flat in the arms that ablate it), and the DV must
    # actually MOVE across arms rather than being bit-identical (the flat-sweep /
    # saturation fingerprint). Reported always; enforced on the smoke.
    by_arm = {r["arm"]: r for r in rows}
    aor_by_arm = {a: float(by_arm[a]["arousal_over_representation_z"])
                  for a in by_arm}
    engagement = {
        "gt_label_varies": all(
            float(r["buffer_std_gt_arousal"]) > GT_AROUSAL_STD_FLOOR for r in rows),
        "strength_route_live_where_expected": all(
            (float(r["memory_strength_range"]) > MS_RANGE_FLOOR) == (r["gain_max"] > 1.0)
            for r in rows),
        "address_route_live_where_expected": all(
            (float(r["retrieval_bias_range"]) > RB_RANGE_FLOOR) == bool(r["tag_on"])
            for r in rows),
        "buffer_alignment_ok": all(bool(r["buffer_gt_alignment_ok"]) for r in rows),
        "dv_varies_across_arms": (
            len({round(v, 9) for v in aor_by_arm.values()}) >= 2),
        "aor_z_by_arm": aor_by_arm,
    }
    print("[smoke] engagement: " + json.dumps(engagement), flush=True)
    if args.dry_run:
        failed = [k for k, v in engagement.items()
                  if isinstance(v, bool) and not v]
        if failed:
            raise AssertionError(
                "dry-run engagement check failed: " + ", ".join(failed)
                + " -- the decisive readout is not properly exercised; fix the driver "
                  "before queuing the full grid"
            )

    aggregate = aggregate_arm_gates([r["arm_gate"] for r in rows])
    criteria_by_arm = {
        ARM_FULL: ["C1_head_authority"],
        ARM_GAIN_FLAT: ["C2_address_route_authority"],
        ARM_TAG_OFF: ["C3_strength_route_authority"],
        ARM_OFF: ["C4_separability"],
    }
    non_degenerate_by_criterion = arm_criteria_non_degenerate(criteria_by_arm, aggregate)

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config: Dict[str, Any] = {
        "threat_env_kwargs": THREAT_ENV_KWARGS,
        "neutral_env_kwargs": NEUTRAL_ENV_KWARGS,
        "arms": ARMS,
        "collect_episodes": COLLECT_EPISODES if not args.dry_run else 8,
        "steps_per_episode": STEPS_PER_EPISODE if not args.dry_run else 20,
        "readout_samples": READOUT_SAMPLES if not args.dry_run else 400,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM,
        "harm_a_dim": HARM_A_DIM,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "bla_retrieval_bias_alpha": BLA_RETRIEVAL_BIAS_ALPHA,
        "thresholds": {
            "MIN_TRACES": MIN_TRACES,
            "GT_AROUSAL_STD_FLOOR": GT_AROUSAL_STD_FLOOR,
            "MS_RANGE_FLOOR": MS_RANGE_FLOOR,
            "RB_RANGE_FLOOR": RB_RANGE_FLOOR,
            "AOR_Z_MARGIN": AOR_Z_MARGIN,
            "OFF_AOR_Z_FLAT_MAX": OFF_AOR_Z_FLAT_MAX,
            "SEEDS_PASS_MIN": SEEDS_PASS_MIN,
        },
        "dry_run": bool(args.dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "backlog_id": "EVB-0074",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": ev["outcome"],
        "evidence_class": "exp:simulation",
        "evidence_direction": ev["evidence_direction"],
        "evidence_direction_per_claim": ev["evidence_direction_per_claim"],
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "experiment_purpose_note": (
            "Tests the PARENT MECH-074 read/write-head SEPARABILITY, not a single "
            "child mechanism. See combination_rule."
        ),
        "acceptance": ev,
        "arm_results": rows,
        "per_seed_results": ev["per_seed"],
        "thresholds": full_config["thresholds"],
        "interpretation": {
            "label": ev["interpretation_label"],
            "preconditions": aggregate["adjudication_preconditions"],
            "criteria_non_degenerate": non_degenerate_by_criterion,
            "combination_rule": ev["combination_rule"],
        },
        "criteria": [
            {"name": "C1_head_authority", "load_bearing": True,
             "passed": bool(ev["c1_head_authority_met"]),
             "claim": "MECH-074"},
            {"name": "C2_address_route_authority", "load_bearing": True,
             "passed": bool(ev["c2_address_route_authority_met"]),
             "claim": "MECH-074b"},
            {"name": "C3_strength_route_authority", "load_bearing": True,
             "passed": bool(ev["c3_strength_route_authority_met"]),
             "claim": "MECH-074a"},
            {"name": "C4_separability", "load_bearing": True,
             "passed": bool(ev["c4_separability_met"]),
             "claim": "MECH-074"},
        ],
        "per_arm_gate": aggregate["per_arm_gate"],
        "engagement_checks": engagement,
        "non_degenerate": bool(aggregate["non_degenerate"]),
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "summary": (
            f"{QUEUE_ID} MECH-074 parent read/write-head route dissociation: "
            f"C1={ev['c1_head_authority_met']} C2={ev['c2_address_route_authority_met']} "
            f"C3={ev['c3_strength_route_authority_met']} C4={ev['c4_separability_met']} "
            f"over {ev['n_seeds']} seed(s); label={ev['interpretation_label']}."
        ),
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(json.dumps({
        "run_id": run_id,
        "outcome": manifest["outcome"],
        "evidence_direction": manifest["evidence_direction"],
        "label": ev["interpretation_label"],
        "manifest": str(out_path),
    }, indent=2))

    _outcome_raw = str(manifest["outcome"]).upper()
    return {
        "outcome": _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        "manifest_path": out_path,
        "dry_run": bool(args.dry_run),
    }


if __name__ == "__main__":
    result = main()
    emit_outcome(
        outcome=result["outcome"],
        manifest_path=result["manifest_path"],
        dry_run=result["dry_run"],
    )
    raise SystemExit(0)
