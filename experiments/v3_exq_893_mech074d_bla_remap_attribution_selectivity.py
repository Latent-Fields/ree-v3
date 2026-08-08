#!/opt/local/bin/python3
"""
V3-EXQ-893 -- MECH-074d BLA PE-remap attribution selectivity (WALL-INDEPENDENT,
              representation-level confirming DV).

Routed from IGW workset item IGW-20260807-235 (GOV-CONFIRM-1). MECH-074d has a BUILT
substrate (SD-035 BLAAnalog, IMPLEMENTED 2026-04-21), literature_confidence 0.809 over
4 literature entries, and ZERO experimental evidence entries of any kind
(claim_evidence.v1.json: genuine_exp_count 0, latest entry a 2026-04-25 lit citation).
The one prior run that tags MECH-074d, V3-EXQ-474, is a MODULE-BOUNDARY substrate-
readiness diagnostic driven by SYNTHETIC PE spikes and hand-supplied candidate dicts;
it never ran the mechanism inside a live agent and never read the representation the
remap is supposed to act on.

WHY A REPRESENTATION DV AND NOT A BEHAVIOURAL ONE (the workset item's own instruction).
A downstream behavioural readout (recall, harm rate, entropy) is confounded by every
other mechanism in the loop, and MECH-074d's own falsifier is stated at the
representation level:

  "EXQ-B must show remap_signal fires on synthetic PE spikes above threshold AND
   preferentially perturbs attributed codes (partial remap, not wholesale); if remap
   fires on sub-threshold PE or perturbs untagged codes uniformly, the attribution
   gate is broken."                                  -- claims.yaml MECH-074d

So the DV here is the CONTENT OF THE STORE THE REMAP WRITES TO --
``agent.e1.context_memory.memory`` -- plus the composition of the target sets the
attribution gate selects. Nothing downstream of the store is read, so no wall
(survival, recall performance, policy entropy) can gate the result.

SUBSTRATE WIRING VERIFIED (/queue-experiment Step 2.5 + 2.5a probe, 2026-08-08).
  ree_core/amygdala/bla.py:472-531   remap gate: PE z-score > sigma AND non-empty
                                     attribution candidates; selects the top
                                     remap_code_fraction of candidates by |contribution|
  ree_core/agent.py:4450-4453        candidates := _get_context_memory_code_contributions
                                     (slot-attention softmax over ContextMemory)
  ree_core/agent.py:4488-4493        fire -> _apply_bla_context_remap
  ree_core/agent.py:3889-3913        the write: mem[i] <- (1-b)*mem[i] + b*write_signal
                                     for each targeted slot i
  ree_core/predictors/e1_deep.py:179 ContextMemory is constructed unconditionally
                                     (16 slots x 128 dims on these encoder dims)

  PROBE FINDING THAT CHANGED THE DESIGN -- ``use_e2_harm_a=True`` IS REQUIRED.
  ``_harm_a_pred_prev`` is produced ONLY when ``agent.e2_harm_a`` exists
  (agent.py:8594), and without a prediction ``bla.tick`` receives
  ``z_harm_a_pred=None`` and remap_signal can NEVER fire. A first probe built exactly
  as the sibling V3-EXQ-888 config (which does not set the flag) recorded
  ``bla._n_remap == 0`` over 296 ticks. With ``use_e2_harm_a=True`` the same rollout
  fired 72 remaps in 137 ticks. This flag is therefore load-bearing for the mechanism
  under test and is set identically in BOTH arms.

  OTHER PROBE READINGS, recorded here because they are what the pre-registered
  thresholds are calibrated AGAINST (they are hypotheses from one short, under-powered
  scratch rollout -- 12 warmup episodes on one seed -- NOT verdicts; settling them at
  power on 3 seeds with a matched control arm is the point of this run):
    - fire fraction ~0.5 of BLA ticks (C4 pre-registers a 0.25 ceiling from normal
      theory: a > 1-SD excursion of a roughly symmetric PE distribution is ~0.16)
    - attribution normalised entropy 1.0000, top-k mass 0.3129 vs a 0.3125 chance
      floor (C1's falsifier: a uniform contribution vector makes "top 33% of
      candidates" an arbitrary index-ordered selection, which is precisely the
      broadcast architecture Moita 2004 rules out)
    - consecutive target-set Jaccard 1.0 -- the SAME five slots every fire
    - ContextMemory slot pairwise cosine similarity driven from mean 0.0139 (no remap)
      to 1.0000 (remap on): total homogenisation of the store, i.e. the opposite of
      the "partial, NOT wholesale replacement" the claim asserts

DESIGN -- 2 arms x 3 distinct seeds [42, 43, 45]. (Seed 44 is excluded per the standing
per-seed early-episode-death instability on this reef-config env family -- EXQ-539-540,
V3-EXQ-538a. 45 substituted.)

  ARM_REMAP_ON   bla_remap_pe_sigma_threshold = 1.0   (the SD-035 default)
  ARM_REMAP_OFF  bla_remap_pe_sigma_threshold = 1e9   (gate can never open)

  REMAP IS DISABLED IN BOTH ARMS DURING P0. This is the load-bearing design decision.
  The remap writes IN PLACE to the very store the DV reads, so if it ran during warmup
  the ON arm would ENTER the measurement window with an already-degraded ContextMemory
  and its readiness precondition (R1, slot differentiation) would fail -- burying the
  real finding as "substrate not ready". Holding the gate shut through P0 means both
  arms start the measurement window from a matched, differentiated store, so every
  difference measured afterwards is attributable to the remap and to nothing else.
  The per-arm threshold is installed on ``agent.bla.config`` at the P0/P1 boundary.

  ARM_REMAP_OFF is NOT decoration: ordinary ContextMemory writes move slots on their
  own, and the write_gate the remap blends toward is the same one the ordinary write
  path uses -- so "targeted slots moved" is NOT by itself evidence of anything. OFF is
  the matched drift control that C3 divides by.

  A BROADCAST-ABLATION ARM (bla_remap_requires_attribution=False) WAS CONSIDERED AND
  DELIBERATELY OMITTED: reading bla.py:504-531, with candidates present -- and the
  agent ALWAYS supplies them, since the softmax puts positive mass on every slot --
  the requires_attribution=False path is bit-identical to the default. It would be an
  inert arm, i.e. exactly the DV-symmetry artifact this skill's design audit exists to
  refuse, not a third condition.

DV-SYMMETRY DECLARATION (one per arm, per the V3-EXQ-604c rule).
  The DV group is {per-slot content of cm.memory, summarised as per-slot displacement
  and as the spread of off-diagonal pairwise cosine similarity} plus {the index
  composition of the remap target sets}. Symmetries these statistics ARE invariant
  under: a permutation of slot indices (the spread statistics are symmetric functions
  over slots), and any transform applied identically to every slot.
  ARM_REMAP_ON  -- the manipulation is enabling an in-place, PER-SLOT blend of a
    SELECTED MINORITY of slots toward a common write_signal. It is neither a
    permutation of slots nor a transform applied uniformly to all of them, so it is
    NOT invariant under the DV's symmetry group; a blend of a strict subset toward a
    common vector strictly contracts pairwise distances among the written slots and
    therefore MUST move the spread statistic. Confirmed live in the probe: exactly 5
    of 16 slots displaced per event, per-event max displacement 0.0045-0.0275.
  ARM_REMAP_OFF -- the manipulation is the ABSENCE of that write (threshold 1e9). The
    DV registers presence-vs-absence of the write directly; probe confirms the store's
    spread is preserved (0.0898) with no remap and collapses (0.0000) with it.
  C1/C2/C4 are measured only in ARM_REMAP_ON and are SCOPED OUT of ARM_REMAP_OFF via
  applies_to/structural_max rather than failed by it (the V3-EXQ-785 rule): with the
  gate shut there are no fires and no target sets, so those statistics are not
  meaningful there -- and that scoping must NOT vacate the ON arm.

CRITERIA (all thresholds are constants below, pre-registered, never derived post-hoc).
  C1 attribution_selectivity  [LOAD-BEARING] mean over fire ticks of (mass on the k
      selected candidates) minus the k/n chance floor > ATTR_MASS_EXCESS_MARGIN.
      Falsifier: a uniform contribution vector -> the "attribution gate" selects an
      arbitrary fixed index set, which is the broadcast architecture Moita 2004
      excludes and which the claim says the substrate MUST NOT have.
  C2 context_differentiated_addressing [LOAD-BEARING] mean WITHIN-context target-set
      Jaccard minus mean CROSS-context Jaccard > CONTEXT_JACCARD_GAP_MARGIN. This is
      the Moita contextual-vs-auditory dissociation transposed onto the substrate: a
      remap addressed to the PREDICTOR codes must target different codes in different
      predictive contexts. Deliberately NOT a bare Jaccard: a bare Jaccard of 1.0 is
      produced BOTH by perfect addressing AND by a constant degenerate target set, so
      it aliases two opposite verdicts. The within-minus-cross gap does not: a
      constant target set gives 1.0 - 1.0 = 0 and fails.
  C3 partial_not_wholesale    [LOAD-BEARING] ON-arm end-of-window slot differentiation
      >= SLOT_DIFF_RATIO_FLOOR x the matched OFF-arm value at the same seed.
      Falsifier: homogenisation of the store = wholesale replacement, the explicit
      negative half of the claim ("Not wholesale replacement").
  C4 pe_spike_sparsity        fraction of BLA ticks carrying a fire <= FIRE_FRAC_CEIL.
      Falsifier: firing on a large share of ticks is not a PE-SPIKE-gated event, and
      accumulates to wholesale rewriting however partial each single fire is.

  PASS iff C1 AND C2 AND C3 AND C4, each met on >= SEEDS_PASS_MIN of the seeds.
  See ``combination_rule`` in the manifest -- a plain AND, recorded explicitly.

READINESS vs VERDICT (why the readiness gate is not circular). The readiness
preconditions assert the INPUTS to the mechanism are non-degenerate; the criteria then
judge the mechanism. R1 asserts the ContextMemory slots are DIFFERENTIATED at the start
of the measurement window -- if every slot were identical the attention softmax would be
uniform BY CONSTRUCTION and C1 could not fail for any reason attributable to the claim
(it would be starved, not falsified). With differentiated slots C1 becomes genuinely
falsifiable: uniform attribution then IS a property of the proxy, not an artifact of an
unexercised store. R1 is therefore strictly upstream of C1 and is measured on a positive
control (a post-P0 agent in the threat context), never on the same statistic C1 routes
on -- which would make C1 unreachable.

SELF-ROUTE. If ARM_REMAP_ON's gate is red the run reports non_degenerate=false and
labels ``substrate_not_ready_requeue`` -- never a substrate verdict -- exactly as the
workset item instructs. A green ON arm yields a real direction on MECH-074d.

experiment_purpose=evidence. claim_ids: MECH-074d only. Wall-independent: every readout
is internal to the ContextMemory store and the BLA gate; no behavioural outcome is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

EXPERIMENT_TYPE = "v3_exq_893_mech074d_bla_remap_attribution_selectivity"
QUEUE_ID = "V3-EXQ-893"
CLAIM_IDS: List[str] = ["MECH-074d"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 excluded (recurring early-episode-death instability on this reef-config env
# family: EXQ-539-540, V3-EXQ-538a). 45 substituted.
SEEDS = [42, 43, 45]

# Encoder dims -- matched to the sibling MECH-074 probe V3-EXQ-888 so the BLA arousal
# calibration and the env kwargs carry over unchanged.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32       # z_harm   (SD-010 sensory)
HARM_A_DIM = 16     # z_harm_a (SD-011 affective) -- the stream the PE is taken on
HARM_HISTORY_LEN = 10

P0_EPISODES = 20        # encoder/world warmup, remap gate held SHUT in both arms
MEASURE_EPISODES = 16    # frozen-encoder measurement window, per-arm gate installed
STEPS_PER_EPISODE = 60
TOTAL_EPISODES = P0_EPISODES + MEASURE_EPISODES   # = episodes_per_run in the queue

# BLA calibration -- identical across arms; ONLY the PE sigma threshold differs.
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5
BLA_REMAP_CODE_FRACTION = 0.33     # SD-035 default (Moita 2004 ~30-35% overwrite)
BLA_CONTEXT_REMAP_BLEND = 0.5      # SD-035 default
REMAP_SIGMA_ON = 1.0               # SD-035 default: ~1 SD of the running PE dist
REMAP_SIGMA_OFF = 1.0e9            # gate can never open
P0_REMAP_SIGMA = 1.0e9             # both arms, during P0 only

# ---- Pre-registered acceptance thresholds (constants; never derived post-hoc) ----
# Readiness floors.
MIN_REMAP_EVENTS = 20         # ON arm: enough fires to estimate the target statistics
SLOT_DIFF_STD_FLOOR = 0.02    # both arms: slots differentiated at window START
PE_SPREAD_FLOOR = 1.0e-6      # both arms: the PE the gate reads must actually vary
MIN_WINDOWS_PER_CONTEXT = 4   # ON arm: both contexts sampled enough for C2

# Verdict thresholds.
# C1: chance mass on k of n candidates is k/n. 0.05 is a 16-percentage-point-relative
# excess on the k/n=0.3125 floor these dims give -- comfortably above the ~0.0004
# the probe measured, and far below what any genuinely selective gate would produce.
ATTR_MASS_EXCESS_MARGIN = 0.05
# C2: a constant target set scores exactly 0.0 on this gap by construction.
CONTEXT_JACCARD_GAP_MARGIN = 0.05
# C3: the ON store must retain at least half the matched control's differentiation.
SLOT_DIFF_RATIO_FLOOR = 0.5
# C4: a > 1-SD excursion of a roughly symmetric PE distribution is ~0.16 of ticks;
# 0.25 is that with generous headroom, pre-registered from normal theory rather than
# tuned to the probe's ~0.5.
FIRE_FRAC_CEIL = 0.25
SEEDS_PASS_MIN = 2            # >= 2/3 seeds

ARM_ON = "ARM_REMAP_ON"
ARM_OFF = "ARM_REMAP_OFF"

ARMS: List[Dict[str, Any]] = [
    {"arm_id": ARM_ON, "remap_sigma": REMAP_SIGMA_ON, "remap_live": True},
    {"arm_id": ARM_OFF, "remap_sigma": REMAP_SIGMA_OFF, "remap_live": False},
]

# Threat context: SD-022 scheduled limb-damage injection drives RELIABLE,
# policy-independent body damage, so ||z_harm_a|| moves for reasons that do not depend
# on the (frozen, untrained) policy finding hazards.
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
# Neutral context: no hazards AND no scheduled injection. Same obs dims, so one agent
# runs in both. The threat/neutral alternation is what gives C2 two predictive
# contexts to dissociate between.
NEUTRAL_ENV_KWARGS: Dict[str, Any] = dict(
    THREAT_ENV_KWARGS, num_hazards=0, scheduled_limb_damage_enabled=False,
)

CTX_THREAT = "threat"
CTX_NEUTRAL = "neutral"


# --------------------------------------------------------------------------------
# Readiness preconditions. Regime-conditioned with applies_to: the fire-dependent ones
# are structurally impossible in ARM_REMAP_OFF (the gate is shut BY DESIGN there, which
# is the manipulation, not a substrate failure), and ANDing them whole-run would vacate
# the ON arm this design exists to measure -- the V3-EXQ-785 defect.
# --------------------------------------------------------------------------------
PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="slot_differentiation_at_window_start",
        description=(
            "Std of the off-diagonal pairwise cosine similarity of "
            "ContextMemory.memory at the START of the measurement window. If the "
            "slots are identical the attention softmax is uniform BY CONSTRUCTION, so "
            "C1 would be starved rather than falsified. Measured after P0 with the "
            "remap gate held shut in BOTH arms, so both enter matched."
        ),
        control="post-P0 agent in the threat context, remap disabled throughout P0",
        threshold=SLOT_DIFF_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="pe_distribution_spread",
        description=(
            "Std of the harm-PE magnitude the BLA gate reads across measurement ticks. "
            "A constant PE makes the sigma gate meaningless in either direction."
        ),
        control="threat/neutral alternation with scheduled limb-damage injection",
        threshold=PE_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="remap_events_sufficient",
        description=(
            "Number of remap fires in the measurement window. C1/C2 are statistics OVER "
            "fires; with none there is nothing to measure."
        ),
        control="bla_remap_pe_sigma_threshold=1.0 with use_e2_harm_a=True supplying "
                "z_harm_a_pred (verified in the 2026-08-08 probe: 72 fires / 137 ticks)",
        threshold=float(MIN_REMAP_EVENTS),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["remap_live"]),
        applies_note=(
            "not meaningful with the gate shut (sigma=1e9): zero fires is the "
            "manipulation in this arm, not a substrate failure"
        ),
        structural_max=lambda ctx: (0.0 if not ctx["remap_live"] else None),
    ),
    PreconditionSpec(
        name="both_contexts_fired",
        description=(
            "Minimum number of fire-bearing episodes in EACH of the threat and neutral "
            "contexts. C2 is a within-minus-cross-context contrast and is undefined if "
            "one context contributed no target sets."
        ),
        control="episodes alternate threat/neutral through the measurement window",
        threshold=float(MIN_WINDOWS_PER_CONTEXT),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["remap_live"]),
        applies_note=(
            "not meaningful with the gate shut: no arm-OFF episode can carry a target "
            "set, by design"
        ),
        structural_max=lambda ctx: (0.0 if not ctx["remap_live"] else None),
    ),
]


# z_goal liveness recording. A fresh agent is built inside each cell, so the
# accumulator is used rather than holding every arm x seed agent alive to the end.
_ZG = ZGoalStreamAccumulator()


def _arm_ctx(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "remap_sigma": float(arm["remap_sigma"]),
        "remap_live": bool(arm["remap_live"]),
    }


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2) -> REEConfig:
    """Config is IDENTICAL across arms. The only arm difference is the PE sigma
    threshold, which is installed on agent.bla.config at the P0/P1 boundary so that
    P0 is bit-comparable across arms."""
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
        # ARC-033 / ARC-058 harm-a forward model. LOAD-BEARING: without it
        # _harm_a_pred_prev stays None and remap_signal can never fire at all
        # (probe 2026-08-08: 0 fires in 296 ticks without it).
        use_e2_harm_a=True,
        # SD-035 amygdala analogue -- BLA on; CeA off (not under test).
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=False,
        bla_arousal_threshold_on=BLA_AROUSAL_THRESHOLD_ON,
        bla_arousal_peak=BLA_AROUSAL_PEAK,
        bla_window_steps=BLA_WINDOW_STEPS,
        bla_remap_pe_sigma_threshold=P0_REMAP_SIGMA,   # shut during P0 in BOTH arms
        bla_remap_code_fraction=BLA_REMAP_CODE_FRACTION,
        bla_remap_requires_attribution=True,
        bla_context_remap_blend=BLA_CONTEXT_REMAP_BLEND,
        replay_diversity_enabled=True,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _slot_diff_std(memory: torch.Tensor) -> float:
    """Spread of the off-diagonal pairwise cosine similarity of the slot matrix.

    This is the DV's differentiation statistic: 0 means every slot points the same way
    (a homogenised store), larger means the slots carry distinguishable content.
    """
    with torch.no_grad():
        m = torch.nn.functional.normalize(memory.detach().to(torch.float32), dim=-1)
        sim = m @ m.t()
        n = sim.shape[0]
        if n < 2:
            return 0.0
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return float(off.std().item())


def _slot_mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-slot cosine similarity between two snapshots of the store."""
    with torch.no_grad():
        x = torch.nn.functional.normalize(a.detach().to(torch.float32), dim=-1)
        y = torch.nn.functional.normalize(b.detach().to(torch.float32), dim=-1)
        return float((x * y).sum(dim=-1).mean().item())


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    union = sa | sb
    if not union:
        return 0.0
    return float(len(sa & sb) / len(union))


def _chance_jaccard(k: int, n: int) -> float:
    """Analytic expected Jaccard of two independent uniform k-subsets of n.

    E|A n B| ~ k^2/n, |A u B| = 2k - |A n B|. Recorded alongside the observed values
    so the C2 gap is interpretable, though C2 itself routes on the within-minus-cross
    contrast and does NOT need this floor.
    """
    if k <= 0 or n <= 0:
        return 0.0
    inter = (k * k) / float(n)
    union = (2.0 * k) - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _run_phase(
    agent: REEAgent,
    threat_env: CausalGridWorldV2,
    neutral_env: CausalGridWorldV2,
    *,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    episode_offset: int,
    total_episodes: int,
    train_mode: bool,
    record: bool,
    label: str,
) -> Dict[str, Any]:
    """Run alternating threat/neutral episodes.

    When record=True, captures per-tick: the exact attribution-contribution vector the
    BLA gate consumed that tick, the remap target set it produced, the harm-PE
    magnitude, and the episode's context label; plus a per-episode snapshot of the
    ContextMemory store.

    The contributions are captured by wrapping ``_get_context_memory_code_contributions``
    so the recorded vector is the one the gate ACTUALLY saw, computed before the remap
    mutated the store on that same tick -- recomputing it after the step would read a
    store the remap had already changed.
    """
    cm = agent.e1.context_memory
    n_slots = int(cm.memory.shape[0])

    captured: Dict[str, Any] = {"contrib": None}
    orig_contrib = agent._get_context_memory_code_contributions

    def _contrib_spy(z_self, z_world):
        out = orig_contrib(z_self, z_world)
        captured["contrib"] = out
        return out

    if record:
        agent._get_context_memory_code_contributions = _contrib_spy

    fire_ticks: List[Dict[str, Any]] = []
    pe_magnitudes: List[float] = []
    n_bla_ticks = 0
    n_no_bla_tick = 0
    episodes_with_fire = {CTX_THREAT: 0, CTX_NEUTRAL: 0}
    per_episode: List[Dict[str, Any]] = []

    if train_mode:
        agent.train()
    else:
        agent.eval()

    try:
        for ep in range(num_episodes):
            is_threat = (ep % 2 == 0)
            ctx = CTX_THREAT if is_threat else CTX_NEUTRAL
            env = threat_env if is_threat else neutral_env
            harness = StepHarness(agent, env, train_mode=train_mode,
                                  seed=seed + episode_offset + ep)
            _, obs_dict = env.reset()
            agent.reset()
            if agent.bla is not None:
                agent.bla.reset()
            harness.reset()

            mem_before = cm.memory.data.detach().clone() if record else None
            ep_fires = 0
            ep_target_sets: List[List[int]] = []

            for _ in range(steps_per_episode):
                # Clear the BLA output latch IMMEDIATELY before the step so a tick on
                # which the BLA did not run is counted as such rather than re-recording
                # the previous tick's remap_signal as a fresh observation.
                agent._bla_last_output = None
                captured["contrib"] = None

                result = harness.step(obs_dict)

                if record:
                    out = agent._bla_last_output
                    if out is None:
                        n_no_bla_tick += 1
                    else:
                        n_bla_ticks += 1
                        pe_magnitudes.append(float(out.pe_magnitude))
                        sig = out.remap_signal or {}
                        if sig:
                            targets = sorted(int(k) for k in sig.keys())
                            contrib = captured["contrib"] or {}
                            vals = np.array(
                                [float(contrib.get(i, 0.0)) for i in range(n_slots)],
                                dtype=np.float64,
                            )
                            total = float(vals.sum())
                            k = len(targets)
                            sel_mass = (
                                float(vals[targets].sum() / total) if total > 1e-12
                                else 0.0
                            )
                            order = np.sort(vals)[::-1]
                            norm_entropy = 0.0
                            if total > 1e-12 and n_slots > 1:
                                p = vals / total
                                nz = p[p > 0]
                                norm_entropy = float(
                                    -(nz * np.log(nz)).sum() / math.log(n_slots)
                                )
                            fire_ticks.append({
                                "context": ctx,
                                "targets": targets,
                                "k": int(k),
                                "n_candidates": int(n_slots),
                                "selected_mass": sel_mass,
                                "chance_mass": float(k) / float(n_slots),
                                "mass_excess": sel_mass - (float(k) / float(n_slots)),
                                "attr_norm_entropy": norm_entropy,
                                "attr_top_value": float(order[0]) if order.size else 0.0,
                                "pe_magnitude": float(out.pe_magnitude),
                                "pe_baseline_std": float(out.pe_baseline_std),
                            })
                            ep_fires += 1
                            ep_target_sets.append(targets)

                obs_dict = result.next_obs_dict
                if result.done:
                    break

            if record:
                mem_after = cm.memory.data.detach().clone()
                disp = (mem_after - mem_before).norm(dim=-1)
                tgt_counts = np.zeros(n_slots, dtype=np.float64)
                for ts in ep_target_sets:
                    for i in ts:
                        tgt_counts[i] += 1.0
                touched = tgt_counts > 0
                d = disp.detach().cpu().numpy().astype(np.float64)
                per_episode.append({
                    "episode": int(ep),
                    "context": ctx,
                    "n_fires": int(ep_fires),
                    "slot_diff_std_end": _slot_diff_std(mem_after),
                    "slot_retention_cos": _slot_mean_cos(mem_before, mem_after),
                    "mean_disp_targeted": (
                        float(d[touched].mean()) if bool(touched.any()) else 0.0),
                    "mean_disp_untargeted": (
                        float(d[~touched].mean()) if bool((~touched).any()) else 0.0),
                    "n_slots_targeted": int(touched.sum()),
                })
                if ep_fires > 0:
                    episodes_with_fire[ctx] += 1

            done_ep = episode_offset + ep + 1
            if done_ep % 5 == 0 or done_ep == total_episodes:
                print(f"  [train] {label} ep {done_ep}/{total_episodes}", flush=True)
    finally:
        if record:
            agent._get_context_memory_code_contributions = orig_contrib

    return {
        "fire_ticks": fire_ticks,
        "pe_magnitudes": pe_magnitudes,
        "n_bla_ticks": int(n_bla_ticks),
        "n_no_bla_tick": int(n_no_bla_tick),
        "episodes_with_fire": episodes_with_fire,
        "per_episode": per_episode,
    }


def _summarise(
    meas: Dict[str, Any],
    *,
    n_slots: int,
    slot_diff_start: float,
    slot_diff_end: float,
    retention_window: float,
) -> Dict[str, Any]:
    fires = meas["fire_ticks"]
    pes = meas["pe_magnitudes"]
    n_bla = int(meas["n_bla_ticks"])

    out: Dict[str, Any] = {
        "n_bla_ticks": n_bla,
        "n_no_bla_tick": int(meas["n_no_bla_tick"]),
        "n_remap_events": len(fires),
        "fire_fraction": (len(fires) / n_bla) if n_bla else 0.0,
        "pe_magnitude_mean": float(np.mean(pes)) if pes else 0.0,
        "pe_magnitude_std": float(np.std(pes)) if pes else 0.0,
        "slot_diff_std_window_start": float(slot_diff_start),
        "slot_diff_std_window_end": float(slot_diff_end),
        "slot_retention_cos_window": float(retention_window),
        "n_slots": int(n_slots),
        "episodes_with_fire_threat": int(meas["episodes_with_fire"][CTX_THREAT]),
        "episodes_with_fire_neutral": int(meas["episodes_with_fire"][CTX_NEUTRAL]),
        # C1 / C2 statistics -- 0.0 when there are no fires, which only ever happens in
        # the gate-shut arm where both are scoped out anyway.
        "attr_mass_excess_mean": 0.0,
        "attr_selected_mass_mean": 0.0,
        "attr_chance_mass": 0.0,
        "attr_norm_entropy_mean": 0.0,
        "target_set_k_mean": 0.0,
        "jaccard_within_context_mean": 0.0,
        "jaccard_cross_context_mean": 0.0,
        "jaccard_context_gap": 0.0,
        "jaccard_chance": 0.0,
        "n_jaccard_within_pairs": 0,
        "n_jaccard_cross_pairs": 0,
    }
    if not fires:
        return out

    out["attr_mass_excess_mean"] = float(np.mean([f["mass_excess"] for f in fires]))
    out["attr_selected_mass_mean"] = float(np.mean([f["selected_mass"] for f in fires]))
    out["attr_chance_mass"] = float(np.mean([f["chance_mass"] for f in fires]))
    out["attr_norm_entropy_mean"] = float(
        np.mean([f["attr_norm_entropy"] for f in fires]))
    ks = [int(f["k"]) for f in fires]
    out["target_set_k_mean"] = float(np.mean(ks))
    out["jaccard_chance"] = _chance_jaccard(int(round(float(np.mean(ks)))), n_slots)

    # C2: within-context vs cross-context target-set overlap. Pairs are capped so a
    # long window does not blow up into O(n^2) work; the cap is applied identically to
    # both pair classes so it cannot bias the gap.
    MAX_PAIRS = 4000
    by_ctx: Dict[str, List[List[int]]] = {CTX_THREAT: [], CTX_NEUTRAL: []}
    for f in fires:
        by_ctx[f["context"]].append(f["targets"])

    within: List[float] = []
    for ctx in (CTX_THREAT, CTX_NEUTRAL):
        sets = by_ctx[ctx]
        for a, b in itertools.islice(itertools.combinations(sets, 2), MAX_PAIRS):
            within.append(_jaccard(a, b))
    cross: List[float] = []
    for a, b in itertools.islice(
        itertools.product(by_ctx[CTX_THREAT], by_ctx[CTX_NEUTRAL]), MAX_PAIRS
    ):
        cross.append(_jaccard(a, b))

    out["n_jaccard_within_pairs"] = len(within)
    out["n_jaccard_cross_pairs"] = len(cross)
    if within:
        out["jaccard_within_context_mean"] = float(np.mean(within))
    if cross:
        out["jaccard_cross_context_mean"] = float(np.mean(cross))
    if within and cross:
        out["jaccard_context_gap"] = (
            out["jaccard_within_context_mean"] - out["jaccard_cross_context_mean"])
    return out


def _run_cell(seed: int, arm: Dict[str, Any], *, dry: bool) -> Dict[str, Any]:
    arm_id = arm["arm_id"]
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    p0_eps = 4 if dry else P0_EPISODES
    meas_eps = 4 if dry else MEASURE_EPISODES
    steps = 20 if dry else STEPS_PER_EPISODE
    total_eps = p0_eps + meas_eps

    config_slice = {
        "arm_id": arm_id,
        "remap_sigma_measurement": float(arm["remap_sigma"]),
        "remap_sigma_p0": float(P0_REMAP_SIGMA),
        "bla_remap_code_fraction": BLA_REMAP_CODE_FRACTION,
        "bla_context_remap_blend": BLA_CONTEXT_REMAP_BLEND,
        "bla_remap_requires_attribution": True,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "use_e2_harm_a": True,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM,
        "harm_a_dim": HARM_A_DIM,
        "p0_episodes": p0_eps,
        "measure_episodes": meas_eps,
        "steps_per_episode": steps,
    }
    label = f"mech074d seed={seed} arm={arm_id}"

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
        # No extra ineligibility reason: each cell builds its OWN envs and agent, and
        # arm_cell resets all RNG on entry, so a cell is a pure function of
        # (substrate, config_slice, seed). The fingerprint is driver-INCLUSIVE (the
        # default), so a same-driver successor can reuse these cells; cross-driver
        # reuse would additionally need this arm's path factored into
        # experiments/_lib/baselines/, which is NOT done here and is recorded as
        # follow-on rather than claimed.
    ) as cell:
        threat_env = _make_env(seed, THREAT_ENV_KWARGS)
        neutral_env = _make_env(seed + 1, NEUTRAL_ENV_KWARGS)
        cfg = _build_config(threat_env)
        agent = REEAgent(cfg)
        cm = agent.e1.context_memory
        n_slots = int(cm.memory.shape[0])

        # ---- P0: warmup with the remap gate held SHUT in BOTH arms. ----
        _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=p0_eps, steps_per_episode=steps, seed=seed,
            episode_offset=0, total_episodes=total_eps,
            train_mode=True, record=False, label=label,
        )

        # ---- Install the arm's gate, then measure with the encoder frozen. ----
        slot_diff_start = _slot_diff_std(cm.memory.data)
        mem_window_start = cm.memory.data.detach().clone()
        if agent.bla is not None:
            agent.bla.config.remap_pe_sigma_threshold = float(arm["remap_sigma"])

        meas = _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=meas_eps, steps_per_episode=steps, seed=seed,
            episode_offset=p0_eps, total_episodes=total_eps,
            train_mode=False, record=True, label=label,
        )
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time

        slot_diff_end = _slot_diff_std(cm.memory.data)
        retention = _slot_mean_cos(mem_window_start, cm.memory.data)

        summ = _summarise(
            meas, n_slots=n_slots,
            slot_diff_start=slot_diff_start, slot_diff_end=slot_diff_end,
            retention_window=retention,
        )

        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            "remap_sigma": float(arm["remap_sigma"]),
            "remap_live": bool(arm["remap_live"]),
            **summ,
            # Generous recording: the full per-episode series, so a successor can
            # re-derive any other displacement/retention statistic without re-running.
            "per_episode": meas["per_episode"],
            # The per-fire records, capped so the manifest stays a sane size while
            # still carrying the raw distribution the criteria summarise.
            "fire_records_sample": meas["fire_ticks"][:400],
            "n_fire_records_total": len(meas["fire_ticks"]),
            "pe_magnitude_sample": [float(x) for x in meas["pe_magnitudes"][:400]],
        }
        cell.stamp(row)

    gate = evaluate_arm_gate(
        arm_id,
        _arm_ctx(arm),
        PRECONDITION_SPECS,
        measured={
            "slot_differentiation_at_window_start": float(
                row["slot_diff_std_window_start"]),
            "pe_distribution_spread": float(row["pe_magnitude_std"]),
            "remap_events_sufficient": float(row["n_remap_events"]),
            "both_contexts_fired": float(min(
                row["episodes_with_fire_threat"], row["episodes_with_fire_neutral"])),
        },
    )
    row["arm_gate"] = gate

    # Per-cell verdict (progress line only; the cohort evaluation is authoritative).
    if arm["remap_live"]:
        cell_pass = bool(
            gate["gate_green"]
            and row["attr_mass_excess_mean"] > ATTR_MASS_EXCESS_MARGIN
            and row["jaccard_context_gap"] > CONTEXT_JACCARD_GAP_MARGIN
            and row["fire_fraction"] <= FIRE_FRAC_CEIL
        )
    else:
        cell_pass = bool(gate["gate_green"] and row["n_remap_events"] == 0)
    print(f"verdict: {'PASS' if cell_pass else 'FAIL'}", flush=True)
    row["cell_pass"] = bool(cell_pass)
    return row


def _evaluate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = r

    per_seed: List[Dict[str, Any]] = []
    c1_ok = c2_ok = c3_ok = c4_ok = 0
    for seed, arms in sorted(by_seed.items()):
        on = arms.get(ARM_ON)
        off = arms.get(ARM_OFF)
        if on is None or off is None:
            continue

        mass_excess = float(on["attr_mass_excess_mean"])
        jac_gap = float(on["jaccard_context_gap"])
        diff_on = float(on["slot_diff_std_window_end"])
        diff_off = float(off["slot_diff_std_window_end"])
        ratio = (diff_on / diff_off) if diff_off > 1e-12 else 0.0
        fire_frac = float(on["fire_fraction"])

        s_c1 = bool(mass_excess > ATTR_MASS_EXCESS_MARGIN)
        s_c2 = bool(jac_gap > CONTEXT_JACCARD_GAP_MARGIN)
        s_c3 = bool(ratio >= SLOT_DIFF_RATIO_FLOOR)
        s_c4 = bool(fire_frac <= FIRE_FRAC_CEIL)
        c1_ok += int(s_c1)
        c2_ok += int(s_c2)
        c3_ok += int(s_c3)
        c4_ok += int(s_c4)

        per_seed.append({
            "seed": seed,
            "attr_mass_excess_on": mass_excess,
            "attr_selected_mass_on": float(on["attr_selected_mass_mean"]),
            "attr_chance_mass_on": float(on["attr_chance_mass"]),
            "attr_norm_entropy_on": float(on["attr_norm_entropy_mean"]),
            "jaccard_within_on": float(on["jaccard_within_context_mean"]),
            "jaccard_cross_on": float(on["jaccard_cross_context_mean"]),
            "jaccard_gap_on": jac_gap,
            "jaccard_chance_on": float(on["jaccard_chance"]),
            "slot_diff_std_end_on": diff_on,
            "slot_diff_std_end_off": diff_off,
            "slot_diff_ratio_on_over_off": ratio,
            "slot_diff_std_start_on": float(on["slot_diff_std_window_start"]),
            "slot_diff_std_start_off": float(off["slot_diff_std_window_start"]),
            "slot_retention_cos_on": float(on["slot_retention_cos_window"]),
            "slot_retention_cos_off": float(off["slot_retention_cos_window"]),
            "fire_fraction_on": fire_frac,
            "n_remap_events_on": int(on["n_remap_events"]),
            "n_remap_events_off": int(off["n_remap_events"]),
            "c1_attribution_selectivity": s_c1,
            "c2_context_differentiated_addressing": s_c2,
            "c3_partial_not_wholesale": s_c3,
            "c4_pe_spike_sparsity": s_c4,
            "seed_pass": bool(s_c1 and s_c2 and s_c3 and s_c4),
        })

    n_seeds = len(per_seed)
    # Well-defined for any seed count: with the pre-registered 3-seed cohort this is
    # exactly SEEDS_PASS_MIN (2 of 3). It only binds differently when fewer seeds ran
    # than were pre-registered (a smoke), where an unreachable criterion would
    # otherwise force a misleading verdict on a run that never had the seeds to test it.
    seeds_needed = min(SEEDS_PASS_MIN, n_seeds) if n_seeds else SEEDS_PASS_MIN
    c1_met = c1_ok >= seeds_needed
    c2_met = c2_ok >= seeds_needed
    c3_met = c3_ok >= seeds_needed
    c4_met = c4_ok >= seeds_needed
    outcome_pass = bool(c1_met and c2_met and c3_met and c4_met)

    # Direction. The two load-bearing halves of MECH-074d are the ATTRIBUTION GATE
    # (C1+C2, Moita 2004's contextual-vs-auditory dissociation) and PARTIALITY
    # (C3+C4, "not wholesale replacement"). Failing either half is a substantive
    # weakening of the claim as stated, not an equivocal reading -- the substrate is
    # built and was exercised.
    gate_half = bool(c1_met and c2_met)
    partial_half = bool(c3_met and c4_met)
    if outcome_pass:
        direction = "supports"
    elif gate_half or partial_half:
        direction = "mixed"
    else:
        direction = "weakens"

    if outcome_pass:
        label = "mech074d_attribution_gated_partial_remap_supports"
    elif not gate_half and not partial_half:
        label = "mech074d_attribution_gate_vacuous_and_remap_wholesale"
    elif not gate_half:
        label = "mech074d_attribution_gate_vacuous_partiality_holds"
    elif not partial_half:
        label = "mech074d_attribution_gated_but_remap_wholesale"
    else:
        label = "mech074d_remap_attribution_inconclusive"

    return {
        "outcome": "PASS" if outcome_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-074d": direction},
        "interpretation_label": label,
        "n_seeds": n_seeds,
        "seeds_needed": int(seeds_needed),
        "c1_attribution_selectivity_met": c1_met,
        "c2_context_differentiated_addressing_met": c2_met,
        "c3_partial_not_wholesale_met": c3_met,
        "c4_pe_spike_sparsity_met": c4_met,
        "c1_seeds_ok": c1_ok,
        "c2_seeds_ok": c2_ok,
        "c3_seeds_ok": c3_ok,
        "c4_seeds_ok": c4_ok,
        "combination_rule": (
            "PASS iff C1 AND C2 AND C3 AND C4 -- a plain AND, no OR-gating. Each "
            "criterion is met when it holds on >= SEEDS_PASS_MIN of the seeds. C1+C2 "
            "are the ATTRIBUTION-GATE half of MECH-074d (Moita 2004 dissociation); "
            "C3+C4 are the PARTIALITY half ('not wholesale replacement'). C1, C2 and "
            "C3 are load-bearing; C4 is a supporting sparsity check. Direction is "
            "'weakens' only when BOTH halves fail, 'mixed' when exactly one does."
        ),
        "per_seed": per_seed,
    }


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    seeds = SEEDS[:1] if args.dry_run else SEEDS

    # Design-time refusal: prove no arm faces a precondition it cannot satisfy in a way
    # that would silently vacate the run (the V3-EXQ-785 check, BEFORE compute).
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
    # committed: the remap must actually fire in the live arm and never in the control,
    # the store must be differentiated at window start, the PE must vary, and the DV
    # must MOVE between arms rather than being bit-identical (the flat-sweep /
    # saturation fingerprint). Reported always; enforced on the smoke.
    on_rows = [r for r in rows if r["remap_live"]]
    off_rows = [r for r in rows if not r["remap_live"]]
    diff_end_by_arm = {r["arm"]: float(r["slot_diff_std_window_end"]) for r in rows}
    engagement = {
        "remap_fires_in_live_arm": all(
            int(r["n_remap_events"]) >= 1 for r in on_rows) if on_rows else False,
        "remap_silent_in_control_arm": all(
            int(r["n_remap_events"]) == 0 for r in off_rows) if off_rows else False,
        "store_differentiated_at_window_start": all(
            float(r["slot_diff_std_window_start"]) > SLOT_DIFF_STD_FLOOR
            for r in rows),
        "pe_varies": all(float(r["pe_magnitude_std"]) > PE_SPREAD_FLOOR for r in rows),
        "bla_ticked": all(int(r["n_bla_ticks"]) > 0 for r in rows),
        "dv_varies_across_arms": (
            len({round(v, 9) for v in diff_end_by_arm.values()}) >= 2),
        "slot_diff_std_end_by_arm": diff_end_by_arm,
        "fire_fraction_by_arm": {
            r["arm"]: float(r["fire_fraction"]) for r in rows},
        "attr_mass_excess_by_arm": {
            r["arm"]: float(r["attr_mass_excess_mean"]) for r in rows},
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
        ARM_ON: [
            "C1_attribution_selectivity",
            "C2_context_differentiated_addressing",
            "C4_pe_spike_sparsity",
        ],
        ARM_OFF: ["C3_partial_not_wholesale"],
    }
    non_degenerate_by_criterion = arm_criteria_non_degenerate(criteria_by_arm, aggregate)

    # Self-route: a red live arm means the mechanism was not exercised well enough to
    # judge the claim, which is a REQUEUE, never a substrate verdict.
    label = ev["interpretation_label"]
    direction = ev["evidence_direction"]
    per_claim = dict(ev["evidence_direction_per_claim"])
    on_gate_green = any(
        bool(r["arm_gate"].get("gate_green")) for r in rows if r["remap_live"])
    if not on_gate_green:
        label = "substrate_not_ready_requeue"
        direction = "unknown"
        per_claim = {"MECH-074d": "unknown"}

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    full_config: Dict[str, Any] = {
        "threat_env_kwargs": THREAT_ENV_KWARGS,
        "neutral_env_kwargs": NEUTRAL_ENV_KWARGS,
        "arms": ARMS,
        "p0_episodes": P0_EPISODES if not args.dry_run else 4,
        "measure_episodes": MEASURE_EPISODES if not args.dry_run else 4,
        "steps_per_episode": STEPS_PER_EPISODE if not args.dry_run else 20,
        "episodes_per_run": TOTAL_EPISODES if not args.dry_run else 8,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "harm_dim": HARM_DIM,
        "harm_a_dim": HARM_A_DIM,
        "use_e2_harm_a": True,
        "bla_arousal_threshold_on": BLA_AROUSAL_THRESHOLD_ON,
        "bla_arousal_peak": BLA_AROUSAL_PEAK,
        "bla_window_steps": BLA_WINDOW_STEPS,
        "bla_remap_code_fraction": BLA_REMAP_CODE_FRACTION,
        "bla_context_remap_blend": BLA_CONTEXT_REMAP_BLEND,
        "bla_remap_requires_attribution": True,
        "remap_sigma_p0_both_arms": P0_REMAP_SIGMA,
        "thresholds": {
            "MIN_REMAP_EVENTS": MIN_REMAP_EVENTS,
            "SLOT_DIFF_STD_FLOOR": SLOT_DIFF_STD_FLOOR,
            "PE_SPREAD_FLOOR": PE_SPREAD_FLOOR,
            "MIN_WINDOWS_PER_CONTEXT": MIN_WINDOWS_PER_CONTEXT,
            "ATTR_MASS_EXCESS_MARGIN": ATTR_MASS_EXCESS_MARGIN,
            "CONTEXT_JACCARD_GAP_MARGIN": CONTEXT_JACCARD_GAP_MARGIN,
            "SLOT_DIFF_RATIO_FLOOR": SLOT_DIFF_RATIO_FLOOR,
            "FIRE_FRAC_CEIL": FIRE_FRAC_CEIL,
            "SEEDS_PASS_MIN": SEEDS_PASS_MIN,
        },
        "dry_run": bool(args.dry_run),
    }

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": ev["outcome"],
        "evidence_class": "exp:simulation",
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "experiment_purpose_note": (
            "Wall-independent representation-level confirming DV for MECH-074d, routed "
            "from IGW-20260807-235 (GOV-CONFIRM-1: built substrate, lit_conf 0.809, "
            "genuine_exp_count 0). Every readout is internal to the ContextMemory store "
            "the remap writes to and to the BLA gate itself; no behavioural outcome is "
            "read, so no downstream wall can gate the result."
        ),
        "acceptance": ev,
        "arm_results": rows,
        "per_seed_results": ev["per_seed"],
        "thresholds": full_config["thresholds"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "criteria_non_degenerate": non_degenerate_by_criterion,
            "combination_rule": ev["combination_rule"],
        },
        "criteria": [
            {"name": "C1_attribution_selectivity", "load_bearing": True,
             "passed": bool(ev["c1_attribution_selectivity_met"]),
             "claim": "MECH-074d"},
            {"name": "C2_context_differentiated_addressing", "load_bearing": True,
             "passed": bool(ev["c2_context_differentiated_addressing_met"]),
             "claim": "MECH-074d"},
            {"name": "C3_partial_not_wholesale", "load_bearing": True,
             "passed": bool(ev["c3_partial_not_wholesale_met"]),
             "claim": "MECH-074d"},
            {"name": "C4_pe_spike_sparsity", "load_bearing": False,
             "passed": bool(ev["c4_pe_spike_sparsity_met"]),
             "claim": "MECH-074d"},
        ],
        "per_arm_gate": aggregate["per_arm_gate"],
        "engagement_checks": engagement,
        "non_degenerate": bool(aggregate["non_degenerate"] and on_gate_green),
        "degeneracy_reason": (
            aggregate["degeneracy_reason"] if aggregate["degeneracy_reason"]
            else ("" if on_gate_green else
                  "ARM_REMAP_ON gate red: the remap mechanism was not exercised well "
                  "enough to judge MECH-074d; self-routed substrate_not_ready_requeue")
        ),
        "summary": (
            f"{QUEUE_ID} MECH-074d BLA remap attribution selectivity: "
            f"C1={ev['c1_attribution_selectivity_met']} "
            f"C2={ev['c2_context_differentiated_addressing_met']} "
            f"C3={ev['c3_partial_not_wholesale_met']} "
            f"C4={ev['c4_pe_spike_sparsity_met']} "
            f"over {ev['n_seeds']} seed(s); label={label}."
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
        "label": label,
        "manifest": str(out_path),
    }, indent=2))
    print(f"Result written to: {out_path}", flush=True)

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
