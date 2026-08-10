#!/opt/local/bin/python3
"""
V3-EXQ-894c -- MECH-074d BLA trainable attribution head: ENTROPY_WEIGHT / MSE BALANCE
               RETUNE (WALL-INDEPENDENT, representation-level DV).

CONFIG/HYPERPARAMETER RETUNE, not an architecture change. Routed from the confirmed
autopsy ``failure_autopsy_V3-EXQ-906a_894b_2026-08-09`` Target 2 (/failure-autopsy ->
/implement-substrate amend, priority 1), which the 2026-08-10 /governance cycle used to
amend ``substrate_queue.json``'s SD-035 entry: the prior failure_record item
(``built_pending_retest``, pointing at V3-EXQ-894b) is superseded by a narrower target --
retune ``entropy_weight``/MSE balance and re-test on the same instrument. This driver is
that re-test. MECH-074d stays provisional/pending_retest_after_substrate.

WHY THIS RUN, AND WHAT IS ALREADY CLOSED.
Three independent experiments have run on this claim:

  V3-EXQ-894   FAIL/weakens. Fixed non-trainable rule. C1/C2 fail 2/3 seeds. Flagged an
               over-firing / dilution confound.
  V3-EXQ-894a  FAIL/weakens. Swept ``bla_remap_pe_sigma_threshold`` over
               [1.0, 1.5, 2.0, 2.5] and REFUTED the confound (Spearman -1.0 on both
               mass_excess and fire_fraction vs sigma). Sigma is CLOSED, held at 1.0.
  V3-EXQ-894b  FAIL/weakens (post-substrate retest of the newly-built
               ``BLAAttributionHead``). ``ARM_HEAD_TRAINED``/``ARM_HEAD_TRAINED_LONG``
               (entropy_weight held at the module's default 0.02) recovered C1 fully
               (0/3 -> 3/3 seeds) but did NOT move C2 (stayed 0/3 seeds; jaccard_gap
               delta vs the fixed rule -0.006 short / -0.025 long -- flat to slightly
               worse with more training). ``training_budget_helps`` was measured False:
               doubling P0+P1 training ticks (914 -> 1998 updates) did not move C2, so
               training BUDGET is not the limiting factor -- this run therefore uses a
               SINGLE training budget (P0 only) and spends the freed compute on entropy_
               weight RESOLUTION instead of a redundant long-training arm.

THE HYPOTHESIS UNDER TEST -- and why it was already named before this run.
``ree_core/amygdala/attribution_head.py`` design decision (3), written when the head was
built and BEFORE 894b ran:

    "The MSE and entropy terms are in DELIBERATE tension, and that tension is what earns
     C2. The entropy penalty pushes `a` toward peaked (C1)... What rules that out is the
     MSE term... Keep entropy_weight small; a large value buys C1 at C2's expense and
     reproduces seed 43 [894a's context-blind-deterministic signature]."

894b measured exactly that predicted shape: entropy_weight=0.02 (the module default) buys
C1 fully and leaves C2 flat. This run holds every other design choice fixed and sweeps
ONLY entropy_weight downward, testing whether a smaller value lets the MSE term's
context-sensitivity through without losing the entropy term's peakedness gain.

H1: at least one swept entropy_weight value reaches C1 AND C2 (AND C3 AND C4) on >= 2/3
    seeds -- the loss-term balance, not trainability per se, was the missing ingredient.
H0 (falsifies the docstring's own diagnosis): no swept value does better than the fixed
    rule on the attribution-gate half (C1 AND C2) -- the defect is deeper than the loss
    balance and MECH-074d needs a different diagnosis than "retune the weighting".

ARMS (6 arms x 3 seeds; sigma FIXED at 1.0 in every ON arm; SINGLE training budget --
P0 only, no extra P1 head-training phase, per the 894b training_budget_helps=False
finding above).

  ARM_REMAP_OFF     sigma 1e9, gate can never open. Drift control / C3 denominator.
                    Unchanged from 894/894a/894b.
  ARM_HEAD_FIXED    legacy non-trainable proxy. WITHIN-RUN replication of 894a/894b's
                    matched control -- if this arm unexpectedly passes the attribution
                    half, 894/894a/894b did not replicate and the whole entropy_weight
                    comparison is void (reported explicitly as fixed_unexpectedly_passed,
                    never silently absorbed).
  ARM_EW_0p02       BLAAttributionHead, entropy_weight=0.02 (the module DEFAULT --
                    replicates 894b's own finding as the sweep's anchor point).
  ARM_EW_0p01       entropy_weight=0.01 (2x smaller).
  ARM_EW_0p005      entropy_weight=0.005 (4x smaller).
  ARM_EW_0p001      entropy_weight=0.001 (20x smaller -- near the entropy term's floor
                    before it can no longer push attention off uniform at all; see the
                    substrate's own ``temperature_min``/``learn_temperature`` machinery,
                    which supplies sharpness independent of the entropy penalty).

WHAT THIS RUN DOES NOT RE-TEST.
The PE-sigma threshold dimension (closed by 894a) and whether the head trains-and-differs-
from-the-fixed-rule at all (closed by 894b: C1 recovers cleanly at the default weight,
so the head's basic mechanics are not in question). Both held fixed. Also NOT re-tested:
training budget (894b's own null result on ``training_budget_helps`` licenses collapsing
to a single P0-only budget here, freeing the compute for four entropy_weight points
instead of two head-training budgets x two head types).

MEASUREMENT (unchanged from 894a/894b -- see 894b for the full derivation; reproduced here
only where it differs).
Alternating threat/neutral episodes. P0 warms the encoder with the gate SHUT in every arm;
the end-of-P0 ContextMemory store is snapshotted and RESTORED at the start of every
measurement episode, identically in all arms, so each episode is an independent replicate
from the same differentiated baseline. The measurement window runs with the encoder frozen
AND the attribution head frozen. Attribution vectors are captured by wrapping
``REEAgent._get_context_memory_code_contributions``, the single entry point BOTH
implementations dispatch through.

  C1 attribution_selectivity  [LOAD-BEARING] mass on the k selected codes minus chance
                              k/n; margin 0.05.
  C2 context_differentiated_addressing  [LOAD-BEARING] within-context minus
                              cross-context Jaccard of the target set; margin 0.05.
  C3 partial_not_wholesale    [LOAD-BEARING] ON-arm end-of-episode slot differentiation
                              as a fraction of the matched OFF control's; floor 0.5.
  C4 pe_spike_sparsity        [not load-bearing] fire fraction <= 0.25.

RUN PASS iff an entropy-weight arm meets C1 AND C2 AND C3 AND C4 on >= 2/3 of its green
seeds. ARM_HEAD_FIXED passing is NOT a pass for this question.

DOSE-RESPONSE READOUT (the direct test of the docstring's claimed direction). Spearman
rank correlation (canonical ``experiments/_lib/stats.spearman`` -- degeneracy-guarded on
the input vector, not the double-argsort'd ranks; see that module's docstring for why the
naive form is unsafe) of entropy_weight against mean attr_mass_excess (C1 statistic,
predicted POSITIVE -- larger entropy_weight buys more peakedness) and against mean
jaccard_context_gap (C2 statistic, predicted NEGATIVE -- larger entropy_weight costs
context sensitivity), computed over the four swept values' arm-level means. This is
reported regardless of pass/fail, exactly as 894a reported the sigma dose-response
regardless of pass/fail -- a monotonic-as-predicted dose-response is evidence for the
mechanism even on a run where no single arm clears the combined C1+C2+C3+C4 bar.

MANDATORY RECORDING FIX (closes a gap the 894b autopsy found). 894b's manifest omitted the
actual ``entropy_weight`` value from its per-arm config, though it varied nothing across
its own arms (all used the module default) so the gap was latent rather than actively
confusing. Every row and every per-arm summary here stamps ``entropy_weight`` explicitly.

A PRECONDITION THIS DESIGN INHERITS (unchanged from 894a/894b).
Under the legacy ContextMemory write path the slot bank homogenises to off-diagonal cosine
1.0000 within ~24 episodes (V3-EXQ-436c). Sixteen identical slots cannot be differentiated
by ANY attribution rule, so without the per-episode restore this run would measure the
write path rather than the head. ``slot_differentiation_at_window_start`` is the
precondition that enforces it; ``episode_restore_effective`` the engagement check that
proves it fired.

ENGAGEMENT CHECKS THAT MATTER MOST HERE (beyond 894a/894b's).
  trainable_heads_trained_past_warmup -- a head still inside its 200-step warmup falls
      back to the LEGACY proxy, silently making an EW arm a duplicate of ARM_HEAD_FIXED.
  baseline_store_matched_across_arms -- the heads' optimisers must not perturb the
      encoder (detached inputs, own parameters); the post-P0 store must be identical
      across arms at each seed.
  attr_stat_varies_across_ew_arms -- the manipulation (entropy_weight) must actually move
      the attribution statistics across the swept values, else the DV is saturated /
      clamped and the sweep is an instrument fingerprint, not a dose-response.

SELF-ROUTE. If NO ON arm has any green (seed, arm) cell the run reports
non_degenerate=false and labels ``substrate_not_ready_requeue`` -- never a substrate
verdict.

experiment_purpose=evidence. claim_ids: MECH-074d only. Wall-independent: every readout is
internal to the ContextMemory store and the BLA gate; no behavioural outcome is read.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

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
from _lib.stats import spearman  # noqa: E402
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

EXPERIMENT_TYPE = "v3_exq_894c_mech074d_bla_entropy_weight_sweep"
QUEUE_ID = "V3-EXQ-894c"
# NOT a supersession. 894/894a/894b are all VALID evidence about the fixed rule and the
# default-weight trainable head -- they are exactly what motivated this retune. Nothing
# about them is invalidated, so nothing is de-weighted.
SUPERSEDES = None
PRIOR_RUNS = [
    "v3_exq_894_mech074d_bla_remap_attribution_selectivity_20260808T005219Z_v3",
    "v3_exq_894a_mech074d_bla_remap_attribution_selectivity_20260808T101157Z_v3",
    "v3_exq_894b_mech074d_bla_trainable_attribution_head_20260809T081623Z_v3",
]
CLAIM_IDS: List[str] = ["MECH-074d"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 excluded (recurring early-episode-death instability on this reef-config env
# family: EXQ-539-540, V3-EXQ-538a). 45 substituted. Unchanged from 894/894a/894b.
SEEDS = [42, 43, 45]

# Encoder dims -- matched to 894/894a/894b so this run is directly comparable.
WORLD_DIM = 32
SELF_DIM = 32
HARM_DIM = 32       # z_harm   (SD-010 sensory)
HARM_A_DIM = 16     # z_harm_a (SD-011 affective) -- the stream the PE is taken on
HARM_HISTORY_LEN = 10

P0_EPISODES = 20        # encoder/world warmup, remap gate held SHUT in every arm
MEASURE_EPISODES = 16    # frozen-encoder measurement window, per-arm gate installed
STEPS_PER_EPISODE = 60
# Single training budget (P0 only) -- 894b measured training_budget_helps=False, so the
# P1 long-training arm from that run is dropped here; the freed compute buys entropy_
# weight RESOLUTION (four values) instead of a second, uninformative training budget.
TOTAL_EPISODES = P0_EPISODES + MEASURE_EPISODES   # = 36, identical for every arm

# BLA calibration -- identical across arms; ONLY entropy_weight (and, in ARM_REMAP_OFF,
# the PE sigma threshold) differs.
BLA_AROUSAL_THRESHOLD_ON = 0.4
BLA_AROUSAL_PEAK = 0.7
BLA_WINDOW_STEPS = 5
BLA_REMAP_CODE_FRACTION = 0.33     # SD-035 default (Moita 2004 ~30-35% overwrite)
BLA_CONTEXT_REMAP_BLEND = 0.5      # SD-035 default
# PE sigma threshold: CLOSED by 894a (Spearman -1.0 on both mass_excess and fire_fraction
# vs sigma). Held fixed at the SD-035 default across every ON arm; not re-swept here.
REMAP_SIGMA_ON = 1.0
REMAP_SIGMA_OFF = 1.0e9            # gate can never open (drift control)
P0_REMAP_SIGMA = 1.0e9             # all arms, during P0 only

HEAD_FIXED = "contribution_threshold"   # legacy non-trainable proxy
HEAD_TRAINABLE = "trainable"            # BLAAttributionHead (landed 2026-08-09)

# Head hyperparameters -- identical across every trainable arm EXCEPT entropy_weight,
# which IS the swept dimension this run exists to resolve.
ATTR_HEAD_WARMUP_STEPS = 200
ATTR_HEAD_LR = 1e-3
ATTR_HEAD_KEY_DIM = 16
ATTR_HEAD_TEMPERATURE_INIT = 0.5
# The module DEFAULT (0.02) is INCLUDED as the sweep's anchor point -- it replicates
# 894b's own measurement under this run's own seeds/env, so the comparison to the lower
# values is within-run rather than against a different experiment's numbers.
ENTROPY_WEIGHT_SWEEP = [0.02, 0.01, 0.005, 0.001]

# ---- Pre-registered acceptance thresholds (constants; never derived post-hoc) ----
MIN_REMAP_EVENTS = 20
SLOT_DIFF_STD_FLOOR = 0.02
PE_SPREAD_FLOOR = 1.0e-6
MIN_WINDOWS_PER_CONTEXT = 4

ATTR_MASS_EXCESS_MARGIN = 0.05
CONTEXT_JACCARD_GAP_MARGIN = 0.05
SLOT_DIFF_RATIO_FLOOR = 0.5
FIRE_FRAC_CEIL = 0.25
SEEDS_PASS_MIN = 2            # >= 2/3 seeds

ARM_OFF = "ARM_REMAP_OFF"
ARM_FIXED = "ARM_HEAD_FIXED"


def _ew_arm_id(ew: float) -> str:
    # e.g. 0.02 -> "ARM_EW_0p02", 0.001 -> "ARM_EW_0p001"
    s = f"{ew:g}".replace(".", "p")
    return f"ARM_EW_{s}"


ARMS: List[Dict[str, Any]] = [
    {"arm_id": ARM_OFF, "remap_sigma": REMAP_SIGMA_OFF, "remap_live": False,
     "head": HEAD_FIXED, "entropy_weight": 0.02},
    {"arm_id": ARM_FIXED, "remap_sigma": REMAP_SIGMA_ON, "remap_live": True,
     "head": HEAD_FIXED, "entropy_weight": 0.02},
] + [
    {"arm_id": _ew_arm_id(ew), "remap_sigma": REMAP_SIGMA_ON, "remap_live": True,
     "head": HEAD_TRAINABLE, "entropy_weight": float(ew)}
    for ew in ENTROPY_WEIGHT_SWEEP
]
ON_ARM_IDS: List[str] = [ARM_FIXED] + [_ew_arm_id(ew) for ew in ENTROPY_WEIGHT_SWEEP]
TRAINED_ARM_IDS: List[str] = [_ew_arm_id(ew) for ew in ENTROPY_WEIGHT_SWEEP]
EW_BY_ARM: Dict[str, float] = {a["arm_id"]: float(a["entropy_weight"]) for a in ARMS}

# Threat context: SD-022 scheduled limb-damage injection drives RELIABLE,
# policy-independent body damage. Identical to 894/894a/894b.
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
NEUTRAL_ENV_KWARGS: Dict[str, Any] = dict(
    THREAT_ENV_KWARGS, num_hazards=0, scheduled_limb_damage_enabled=False,
)

CTX_THREAT = "threat"
CTX_NEUTRAL = "neutral"


PRECONDITION_SPECS: List[PreconditionSpec] = [
    PreconditionSpec(
        name="slot_differentiation_at_window_start",
        description=(
            "Std of the off-diagonal pairwise cosine similarity of "
            "ContextMemory.memory in the BASELINE store every measurement episode is "
            "restored to."
        ),
        control="post-P0 agent in the threat context, remap disabled throughout P0",
        threshold=SLOT_DIFF_STD_FLOOR,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="pe_distribution_spread",
        description=(
            "Std of the harm-PE magnitude the BLA gate reads across measurement ticks."
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
        control="bla_remap_pe_sigma_threshold=1.0 with use_e2_harm_a=True",
        threshold=float(MIN_REMAP_EVENTS - 1),
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
        threshold=float(MIN_WINDOWS_PER_CONTEXT - 1),
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


_ZG = ZGoalStreamAccumulator()


def _arm_ctx(arm: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "arm_id": arm["arm_id"],
        "remap_sigma": float(arm["remap_sigma"]),
        "remap_live": bool(arm["remap_live"]),
        "head": str(arm["head"]),
        "entropy_weight": float(arm["entropy_weight"]),
    }


def _make_env(seed: int, kwargs: Dict[str, Any]) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **kwargs)


def _build_config(env: CausalGridWorldV2, arm: Dict[str, Any]) -> REEConfig:
    """Config is identical across arms EXCEPT bla_attribution_head and
    bla_attr_head_entropy_weight -- the manipulation -- which must be set at construction
    time (the head is instantiated in REEAgent.__init__ and cannot be swapped in after)."""
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
        use_e2_harm_a=True,
        use_amygdala_analog=True,
        use_bla_analog=True,
        use_cea_analog=False,
        bla_arousal_threshold_on=BLA_AROUSAL_THRESHOLD_ON,
        bla_arousal_peak=BLA_AROUSAL_PEAK,
        bla_window_steps=BLA_WINDOW_STEPS,
        bla_remap_pe_sigma_threshold=P0_REMAP_SIGMA,   # shut during P0 in every arm
        bla_remap_code_fraction=BLA_REMAP_CODE_FRACTION,
        bla_remap_requires_attribution=True,
        bla_context_remap_blend=BLA_CONTEXT_REMAP_BLEND,
        # ---- THE MANIPULATION ----
        bla_attribution_head=str(arm["head"]),
        bla_attr_head_warmup_steps=ATTR_HEAD_WARMUP_STEPS,
        bla_attr_head_lr=ATTR_HEAD_LR,
        bla_attr_head_key_dim=ATTR_HEAD_KEY_DIM,
        bla_attr_head_entropy_weight=float(arm["entropy_weight"]),
        bla_attr_head_temperature_init=ATTR_HEAD_TEMPERATURE_INIT,
        bla_attr_head_train=True,
        replay_diversity_enabled=True,
    )
    cfg.residue.valence_enabled = True
    return cfg


def _slot_diff_std(memory: torch.Tensor) -> float:
    with torch.no_grad():
        m = torch.nn.functional.normalize(memory.detach().to(torch.float32), dim=-1)
        sim = m @ m.t()
        n = sim.shape[0]
        if n < 2:
            return 0.0
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        return float(off.std().item())


def _slot_mean_cos(a: torch.Tensor, b: torch.Tensor) -> float:
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
    restore_base: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Run alternating threat/neutral episodes. Unchanged from 894b -- see that driver's
    docstring for the full derivation of the restore-per-episode design."""
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

            if restore_base is not None:
                with torch.no_grad():
                    cm.memory.data.copy_(restore_base)

            mem_before = cm.memory.data.detach().clone() if record else None
            ep_fires = 0
            ep_target_sets: List[List[int]] = []

            for _ in range(steps_per_episode):
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
                    "slot_diff_std_start": _slot_diff_std(mem_before),
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
    slot_diff_baseline: float,
) -> Dict[str, Any]:
    fires = meas["fire_ticks"]
    pes = meas["pe_magnitudes"]
    n_bla = int(meas["n_bla_ticks"])
    eps = meas["per_episode"]

    diff_end_mean = float(np.mean([e["slot_diff_std_end"] for e in eps])) if eps else 0.0
    diff_start_mean = (
        float(np.mean([e["slot_diff_std_start"] for e in eps])) if eps else 0.0)
    retention_mean = (
        float(np.mean([e["slot_retention_cos"] for e in eps])) if eps else 0.0)

    out: Dict[str, Any] = {
        "n_bla_ticks": n_bla,
        "n_no_bla_tick": int(meas["n_no_bla_tick"]),
        "n_remap_events": len(fires),
        "fire_fraction": (len(fires) / n_bla) if n_bla else 0.0,
        "pe_magnitude_mean": float(np.mean(pes)) if pes else 0.0,
        "pe_magnitude_std": float(np.std(pes)) if pes else 0.0,
        "slot_diff_std_baseline": float(slot_diff_baseline),
        "slot_diff_std_episode_start_mean": diff_start_mean,
        "slot_diff_std_episode_end_mean": diff_end_mean,
        "slot_retention_cos_episode_mean": retention_mean,
        "n_measure_episodes": len(eps),
        "n_slots": int(n_slots),
        "episodes_with_fire_threat": int(meas["episodes_with_fire"][CTX_THREAT]),
        "episodes_with_fire_neutral": int(meas["episodes_with_fire"][CTX_NEUTRAL]),
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
    ew = float(arm["entropy_weight"])
    print(f"Seed {seed} Condition {arm_id}", flush=True)

    p0_eps = 3 if dry else P0_EPISODES
    meas_eps = (2 * MIN_WINDOWS_PER_CONTEXT) if dry else MEASURE_EPISODES
    steps = 15 if dry else STEPS_PER_EPISODE
    total_eps = p0_eps + meas_eps

    config_slice = {
        "arm_id": arm_id,
        "attribution_head": str(arm["head"]),
        "entropy_weight": ew,
        "attr_head_warmup_steps": ATTR_HEAD_WARMUP_STEPS,
        "attr_head_lr": ATTR_HEAD_LR,
        "attr_head_key_dim": ATTR_HEAD_KEY_DIM,
        "attr_head_temperature_init": ATTR_HEAD_TEMPERATURE_INIT,
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
    label = f"mech074d seed={seed} arm={arm_id} ew={ew}"

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        threat_env = _make_env(seed, THREAT_ENV_KWARGS)
        neutral_env = _make_env(seed + 1, NEUTRAL_ENV_KWARGS)
        cfg = _build_config(threat_env, arm)
        agent = REEAgent(cfg)
        cm = agent.e1.context_memory
        n_slots = int(cm.memory.shape[0])

        # ---- P0: warmup with the remap gate held SHUT in every arm. ----
        _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=p0_eps, steps_per_episode=steps, seed=seed,
            episode_offset=0, total_episodes=total_eps,
            train_mode=True, record=False, label=label,
        )

        # ---- Install the arm's gate, then measure with the encoder frozen. ----
        mem_baseline = cm.memory.data.detach().clone()
        slot_diff_baseline = _slot_diff_std(mem_baseline)
        if agent.bla is not None:
            agent.bla.config.remap_pe_sigma_threshold = float(arm["remap_sigma"])

        head_diag_pre_measure: Dict[str, Any] = {}
        if agent.bla_attribution_head is not None:
            head_diag_pre_measure = dict(agent.bla_attribution_head.diagnostics)
            agent.bla_attribution_head.config.train = False

        meas = _run_phase(
            agent, threat_env, neutral_env,
            num_episodes=meas_eps, steps_per_episode=steps, seed=seed,
            episode_offset=p0_eps, total_episodes=total_eps,
            train_mode=False, record=True, label=label,
            restore_base=mem_baseline,
        )
        _ZG.observe(agent)

        summ = _summarise(
            meas, n_slots=n_slots, slot_diff_baseline=slot_diff_baseline,
        )

        head_diag_post = (
            dict(agent.bla_attribution_head.diagnostics)
            if agent.bla_attribution_head is not None else {})

        row: Dict[str, Any] = {
            "seed": int(seed),
            "arm": arm_id,
            "remap_sigma": float(arm["remap_sigma"]),
            "remap_live": bool(arm["remap_live"]),
            "attribution_head": str(arm["head"]),
            # ---- MANDATORY recording fix (894b autopsy gap): the actual entropy_weight
            # value this cell trained with, at both row level and inside config_slice
            # (already stamped above) and inside arm_fingerprint's config_slice.
            "entropy_weight": ew,
            "head_is_trainable": bool(agent.bla_attribution_head is not None),
            "head_diagnostics_pre_measure": head_diag_pre_measure,
            "head_diagnostics_post_measure": head_diag_post,
            "head_n_updates": int(head_diag_post.get("n_updates", 0)),
            "head_is_warm": bool(head_diag_post.get("is_warm", False)),
            "head_warmup_fallbacks": int(head_diag_post.get("n_warmup_fallbacks", 0)),
            "head_norm_entropy": float(
                head_diag_pre_measure.get("last_norm_entropy", 0.0)),
            "head_pred_loss": float(head_diag_pre_measure.get("last_pred_loss", 0.0)),
            "head_max_weight_at_measure": float(
                head_diag_post.get("last_max_weight", 0.0)),
            **summ,
            "per_episode": meas["per_episode"],
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
                row["slot_diff_std_baseline"]),
            "pe_distribution_spread": float(row["pe_magnitude_std"]),
            "remap_events_sufficient": float(row["n_remap_events"]),
            "both_contexts_fired": float(min(
                row["episodes_with_fire_threat"], row["episodes_with_fire_neutral"])),
        },
    )
    row["arm_gate"] = gate

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
    """Per-entropy_weight-arm evaluation, generalising 894b's fixed-vs-trained contrast to
    an N-point sweep. For each ON arm, tally C1-C4 over that arm's GREEN seeds against the
    matched OFF drift control at the same seed; an arm PASSES iff all four are met on
    >= seeds_needed green seeds. RUN PASS iff any entropy-weight arm passes -- ARM_HEAD_
    FIXED passing would mean the within-run replication of 894/894a/894b's failure did not
    reproduce and the comparison is void (flagged separately, never silently absorbed)."""
    by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), {})[r["arm"]] = r

    n_seeds_run = len(by_seed)
    seeds_needed = min(SEEDS_PASS_MIN, n_seeds_run) if n_seeds_run else SEEDS_PASS_MIN

    per_arm_per_seed: List[Dict[str, Any]] = []
    per_arm: List[Dict[str, Any]] = []
    for arm_id in ON_ARM_IDS:
        sigma = REMAP_SIGMA_ON
        ew = EW_BY_ARM[arm_id]
        head = HEAD_TRAINABLE if arm_id in TRAINED_ARM_IDS else HEAD_FIXED
        green_seeds: List[int] = []
        c1_ok = c2_ok = c3_ok = c4_ok = 0
        arm_mass_excess: List[float] = []
        arm_fire_frac: List[float] = []
        arm_jac_gap: List[float] = []
        arm_ratio: List[float] = []
        for seed, arms in sorted(by_seed.items()):
            on = arms.get(arm_id)
            off = arms.get(ARM_OFF)
            if on is None or off is None:
                continue
            gate_green = bool(on.get("arm_gate", {}).get("gate_green"))

            mass_excess = float(on["attr_mass_excess_mean"])
            jac_gap = float(on["jaccard_context_gap"])
            diff_on = float(on["slot_diff_std_episode_end_mean"])
            diff_off = float(off["slot_diff_std_episode_end_mean"])
            ratio = (diff_on / diff_off) if diff_off > 1e-12 else 0.0
            fire_frac = float(on["fire_fraction"])

            s_c1 = bool(mass_excess > ATTR_MASS_EXCESS_MARGIN)
            s_c2 = bool(jac_gap > CONTEXT_JACCARD_GAP_MARGIN)
            s_c3 = bool(ratio >= SLOT_DIFF_RATIO_FLOOR)
            s_c4 = bool(fire_frac <= FIRE_FRAC_CEIL)

            if gate_green:
                green_seeds.append(seed)
                c1_ok += int(s_c1); c2_ok += int(s_c2)
                c3_ok += int(s_c3); c4_ok += int(s_c4)
                arm_mass_excess.append(mass_excess)
                arm_fire_frac.append(fire_frac)
                arm_jac_gap.append(jac_gap)
                arm_ratio.append(ratio)

            per_arm_per_seed.append({
                "arm": arm_id,
                "remap_sigma": sigma,
                "attribution_head": head,
                "entropy_weight": ew,
                "head_n_updates": int(on.get("head_n_updates", 0)),
                "head_is_warm": bool(on.get("head_is_warm", False)),
                "head_norm_entropy": float(on.get("head_norm_entropy", 0.0)),
                "seed": seed,
                "gate_green": gate_green,
                "attr_mass_excess_on": mass_excess,
                "attr_selected_mass_on": float(on["attr_selected_mass_mean"]),
                "attr_chance_mass_on": float(on["attr_chance_mass"]),
                "attr_norm_entropy_on": float(on["attr_norm_entropy_mean"]),
                "jaccard_within_on": float(on["jaccard_within_context_mean"]),
                "jaccard_cross_on": float(on["jaccard_cross_context_mean"]),
                "jaccard_gap_on": jac_gap,
                "jaccard_chance_on": float(on["jaccard_chance"]),
                "slot_diff_std_episode_end_mean_on": diff_on,
                "slot_diff_std_episode_end_mean_off": diff_off,
                "slot_diff_ratio_on_over_off": ratio,
                "slot_diff_std_baseline_on": float(on["slot_diff_std_baseline"]),
                "slot_diff_std_baseline_off": float(off["slot_diff_std_baseline"]),
                "fire_fraction_on": fire_frac,
                "n_remap_events_on": int(on["n_remap_events"]),
                "n_remap_events_off": int(off["n_remap_events"]),
                "c1_attribution_selectivity": s_c1,
                "c2_context_differentiated_addressing": s_c2,
                "c3_partial_not_wholesale": s_c3,
                "c4_pe_spike_sparsity": s_c4,
                "seed_pass": bool(gate_green and s_c1 and s_c2 and s_c3 and s_c4),
            })

        n_green = len(green_seeds)
        eligible = n_green >= seeds_needed
        c1_met = bool(eligible and c1_ok >= seeds_needed)
        c2_met = bool(eligible and c2_ok >= seeds_needed)
        c3_met = bool(eligible and c3_ok >= seeds_needed)
        c4_met = bool(eligible and c4_ok >= seeds_needed)
        arm_pass = bool(c1_met and c2_met and c3_met and c4_met)
        per_arm.append({
            "arm": arm_id,
            "remap_sigma": sigma,
            "attribution_head": head,
            "entropy_weight": ew,
            "is_trainable_arm": bool(arm_id in TRAINED_ARM_IDS),
            "n_green_seeds": n_green,
            "green_seeds": list(green_seeds),
            "c1_seeds_ok": c1_ok, "c2_seeds_ok": c2_ok,
            "c3_seeds_ok": c3_ok, "c4_seeds_ok": c4_ok,
            "c1_attribution_selectivity_met": c1_met,
            "c2_context_differentiated_addressing_met": c2_met,
            "c3_partial_not_wholesale_met": c3_met,
            "c4_pe_spike_sparsity_met": c4_met,
            "arm_pass": arm_pass,
            "attribution_gate_half_met": bool(c1_met and c2_met),
            "partiality_half_met": bool(c3_met and c4_met),
            "mean_attr_mass_excess": (
                float(np.mean(arm_mass_excess)) if arm_mass_excess else None),
            "mean_fire_fraction": (
                float(np.mean(arm_fire_frac)) if arm_fire_frac else None),
            "mean_jaccard_gap": (
                float(np.mean(arm_jac_gap)) if arm_jac_gap else None),
            "mean_slot_diff_ratio": (
                float(np.mean(arm_ratio)) if arm_ratio else None),
        })

    by_arm = {a["arm"]: a for a in per_arm}
    fixed = by_arm.get(ARM_FIXED)

    def _delta(arm_id: str, key: str) -> Optional[float]:
        a = by_arm.get(arm_id)
        if a is None or fixed is None:
            return None
        av, fv = a.get(key), fixed.get(key)
        if av is None or fv is None:
            return None
        return float(av) - float(fv)

    head_contrast = {
        arm_id: {
            "entropy_weight": EW_BY_ARM[arm_id],
            "mass_excess_delta_vs_fixed": _delta(arm_id, "mean_attr_mass_excess"),
            "jaccard_gap_delta_vs_fixed": _delta(arm_id, "mean_jaccard_gap"),
            "fire_fraction_delta_vs_fixed": _delta(arm_id, "mean_fire_fraction"),
            "slot_diff_ratio_delta_vs_fixed": _delta(arm_id, "mean_slot_diff_ratio"),
            "c1_seeds_ok": by_arm[arm_id]["c1_seeds_ok"] if arm_id in by_arm else 0,
            "c2_seeds_ok": by_arm[arm_id]["c2_seeds_ok"] if arm_id in by_arm else 0,
        }
        for arm_id in TRAINED_ARM_IDS if arm_id in by_arm
    }

    fixed_replicates_894 = bool(
        fixed is not None and fixed["n_green_seeds"] >= seeds_needed
        and not fixed["attribution_gate_half_met"])
    fixed_unexpectedly_passed = bool(
        fixed is not None and fixed["attribution_gate_half_met"])

    # ---- Dose-response: does entropy_weight move C1/C2 in the docstring-predicted
    # direction? Positive Spearman for mass_excess (larger weight -> more peaked),
    # negative for jaccard_gap (larger weight -> less context-sensitive). Computed over
    # the swept arms' means, using the degeneracy-guarded canonical helper.
    dr_ew: List[float] = []
    dr_mass: List[float] = []
    dr_jac: List[float] = []
    for arm_id in TRAINED_ARM_IDS:
        a = by_arm.get(arm_id)
        if a is None or a["mean_attr_mass_excess"] is None or a["mean_jaccard_gap"] is None:
            continue
        dr_ew.append(EW_BY_ARM[arm_id])
        dr_mass.append(float(a["mean_attr_mass_excess"]))
        dr_jac.append(float(a["mean_jaccard_gap"]))
    spearman_mass_vs_ew = spearman(dr_ew, dr_mass) if len(dr_ew) >= 2 else None
    spearman_jaccard_vs_ew = spearman(dr_ew, dr_jac) if len(dr_ew) >= 2 else None
    dose_response_matches_docstring = (
        spearman_mass_vs_ew is not None and spearman_jaccard_vs_ew is not None
        and spearman_mass_vs_ew > 0.0 and spearman_jaccard_vs_ew < 0.0
    )

    trained_arms = [a for a in per_arm if a["arm"] in TRAINED_ARM_IDS]
    any_trained_gate_half = any(a["attribution_gate_half_met"] for a in trained_arms)
    any_gate_half = any(a["attribution_gate_half_met"] for a in per_arm)
    any_partial_half = any(a["partiality_half_met"] for a in per_arm)

    passing_arms = [a for a in per_arm
                    if a["arm_pass"] and a["arm"] in TRAINED_ARM_IDS]
    outcome_pass = bool(passing_arms)
    fixed_arm_passed = bool(fixed is not None and fixed["arm_pass"])

    # Best arm: prefer a full pass at the LOWEST entropy_weight that achieves it (a
    # smaller weight is the more conservative / more MSE-driven operating point, and the
    # docstring's own reasoning favours the smallest value that still works). Else the
    # trained arm with the most criteria met; else any scored arm.
    scored = [a for a in per_arm if a["n_green_seeds"] >= 1]
    if passing_arms:
        best_arm = min(passing_arms, key=lambda a: a["entropy_weight"])
    elif trained_arms:
        best_arm = max(
            trained_arms,
            key=lambda a: (a["c1_seeds_ok"] + a["c2_seeds_ok"], -a["entropy_weight"]),
        )
    else:
        best_arm = scored[0] if scored else None

    c1_met_any = any(a["c1_attribution_selectivity_met"] for a in per_arm)
    c2_met_any = any(a["c2_context_differentiated_addressing_met"] for a in per_arm)
    c3_met_any = any(a["c3_partial_not_wholesale_met"] for a in per_arm)
    c4_met_any = any(a["c4_pe_spike_sparsity_met"] for a in per_arm)

    beats_fixed_on_both = any(
        (hc["mass_excess_delta_vs_fixed"] or 0.0) > 0.0
        and (hc["jaccard_gap_delta_vs_fixed"] or 0.0) > 0.0
        for hc in head_contrast.values()
    )
    attribution_recovers = bool(any_trained_gate_half or beats_fixed_on_both)
    if outcome_pass:
        direction = "supports"
        label = (
            "mech074d_entropy_weight_retune_recovers_attribution_gate_"
            f"ew{best_arm['entropy_weight']:g}".replace(".", "p")
        )
    elif attribution_recovers:
        direction = "mixed"
        label = "mech074d_entropy_weight_retune_improves_attribution_no_full_pass"
    else:
        direction = "weakens"
        label = "mech074d_entropy_weight_retune_does_not_recover_attribution_gate"

    combination_rule = (
        "ARM AXIS = entropy_weight (the BLA attribution head's entropy-penalty weight), "
        f"swept over {ENTROPY_WEIGHT_SWEEP} at a FIXED bla_remap_pe_sigma_threshold of "
        f"{REMAP_SIGMA_ON} and a SINGLE P0-only training budget (894b already closed both "
        "the sigma dimension and the training-budget dimension). Arms: ARM_HEAD_FIXED "
        "(legacy non-trainable proxy, a within-run replication of 894/894a/894b's matched "
        "control) and one ARM_EW_<value> per swept entropy_weight (BLAAttributionHead). "
        "For EACH ON arm, criterion Ck is MET when it holds on >= seeds_needed of that "
        "arm's GREEN (gate-passing) seeds; the arm PASSES iff C1 AND C2 AND C3 AND C4. "
        "RUN PASS iff any entropy-weight arm passes -- ARM_HEAD_FIXED passing is NOT a "
        "pass for this question and instead voids the comparison (reported as "
        "fixed_unexpectedly_passed). C1+C2 are the ATTRIBUTION-GATE half (Moita 2004 "
        "dissociation); C3+C4 the PARTIALITY half. The dose-response block reports "
        "Spearman(entropy_weight, mass_excess) and Spearman(entropy_weight, jaccard_gap) "
        "over the swept arms' means, independent of pass/fail, as the direct test of the "
        "module docstring's claimed monotone tradeoff direction."
    )

    return {
        "outcome": "PASS" if outcome_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_per_claim": {"MECH-074d": direction},
        "interpretation_label": label,
        "n_seeds": n_seeds_run,
        "seeds_needed": int(seeds_needed),
        "c1_attribution_selectivity_met": c1_met_any,
        "c2_context_differentiated_addressing_met": c2_met_any,
        "c3_partial_not_wholesale_met": c3_met_any,
        "c4_pe_spike_sparsity_met": c4_met_any,
        "outcome_pass": outcome_pass,
        "passing_arm_ids": [a["arm"] for a in passing_arms],
        "best_arm_id": best_arm["arm"] if best_arm else None,
        "best_arm_head": best_arm["attribution_head"] if best_arm else None,
        "best_entropy_weight": best_arm["entropy_weight"] if best_arm else None,
        "attribution_recovers": attribution_recovers,
        "any_attribution_gate_half_met": any_gate_half,
        "any_trained_attribution_gate_half_met": any_trained_gate_half,
        "any_partiality_half_met": any_partial_half,
        "fixed_arm_passed": fixed_arm_passed,
        "fixed_replicates_894": fixed_replicates_894,
        "fixed_unexpectedly_passed": fixed_unexpectedly_passed,
        "trained_beats_fixed_on_both_attribution_stats": beats_fixed_on_both,
        "spearman_mass_excess_vs_entropy_weight": spearman_mass_vs_ew,
        "spearman_jaccard_gap_vs_entropy_weight": spearman_jaccard_vs_ew,
        "dose_response_matches_docstring_direction": dose_response_matches_docstring,
        "head_contrast": head_contrast,
        "per_arm": per_arm,
        "combination_rule": combination_rule,
        "per_seed": per_arm_per_seed,
    }


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    seeds = SEEDS[:1] if args.dry_run else SEEDS

    assert_no_structurally_unsatisfiable_gate(
        PRECONDITION_SPECS, [_arm_ctx(a) for a in ARMS]
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm in ARMS:
            rows.append(_run_cell(seed, arm, dry=args.dry_run))

    ev = _evaluate(rows)

    on_rows = [r for r in rows if r["remap_live"]]
    off_rows = [r for r in rows if not r["remap_live"]]
    fixed_rows = [r for r in on_rows if r["arm"] == ARM_FIXED]
    trained_rows = [r for r in on_rows if r["arm"] in TRAINED_ARM_IDS]
    diff_end_by_arm = {
        r["arm"]: float(r["slot_diff_std_episode_end_mean"]) for r in rows}
    fire_fraction_by_arm = {r["arm"]: float(r["fire_fraction"]) for r in rows}
    mass_excess_by_arm = {r["arm"]: float(r["attr_mass_excess_mean"]) for r in rows}
    jaccard_gap_by_arm = {r["arm"]: float(r["jaccard_context_gap"]) for r in rows}
    on_mass_excess = [float(r["attr_mass_excess_mean"]) for r in on_rows]

    heads_warm = all(bool(r.get("head_is_warm")) for r in trained_rows) if trained_rows \
        else False
    heads_trained_enough = all(
        int(r.get("head_n_updates", 0)) > ATTR_HEAD_WARMUP_STEPS for r in trained_rows
    ) if trained_rows else False
    fixed_has_no_head = all(
        not bool(r.get("head_is_trainable")) for r in rows if r["arm"] not in
        TRAINED_ARM_IDS)

    baseline_by_seed: Dict[int, List[float]] = {}
    for r in rows:
        baseline_by_seed.setdefault(int(r["seed"]), []).append(
            float(r["slot_diff_std_baseline"]))
    baseline_matched_across_arms = all(
        (max(v) - min(v)) < 1e-9 for v in baseline_by_seed.values())

    # Per-entropy_weight-arm mass_excess, restricted to the swept arms only, for the
    # anti-saturation check (generalises 894b's two-value check to N values).
    ew_mass_excess_values = [mass_excess_by_arm[a] for a in TRAINED_ARM_IDS
                             if a in mass_excess_by_arm]
    ew_jaccard_gap_values = [jaccard_gap_by_arm[a] for a in TRAINED_ARM_IDS
                             if a in jaccard_gap_by_arm]

    engagement = {
        # -- Tier 1: scale-invariant, ENFORCED on --dry-run --
        "remap_fires_in_fixed_arm": all(
            int(r["n_remap_events"]) >= 1 for r in fixed_rows) if fixed_rows else False,
        "remap_fires_in_trained_arms": all(
            int(r["n_remap_events"]) >= 1 for r in trained_rows) if trained_rows
            else False,
        "remap_silent_in_control_arm": all(
            int(r["n_remap_events"]) == 0 for r in off_rows) if off_rows else False,
        "store_differentiated_at_window_start": all(
            float(r["slot_diff_std_baseline"]) > SLOT_DIFF_STD_FLOOR for r in rows),
        "episode_restore_effective": all(
            abs(float(r["slot_diff_std_episode_start_mean"])
                - float(r["slot_diff_std_baseline"])) < 1e-6
            for r in rows),
        "pe_varies": all(float(r["pe_magnitude_std"]) > PE_SPREAD_FLOOR for r in rows),
        "bla_ticked": all(int(r["n_bla_ticks"]) > 0 for r in rows),
        "dv_varies_across_arms": (
            len({round(v, 9) for v in diff_end_by_arm.values()}) >= 2),
        "trainable_heads_are_warm": heads_warm,
        "fixed_arm_built_no_head": fixed_has_no_head,
        # -- Tier 2: REPORTED always, enforced only on a full run --
        "attr_mass_excess_varies_across_ew_arms": (
            len({round(f, 6) for f in ew_mass_excess_values}) >= 2),
        "attr_jaccard_gap_varies_across_ew_arms": (
            len({round(f, 6) for f in ew_jaccard_gap_values}) >= 2),
        "trainable_heads_trained_past_warmup": heads_trained_enough,
        "baseline_store_matched_across_arms": baseline_matched_across_arms,
        "slot_diff_std_end_by_arm": diff_end_by_arm,
        "fire_fraction_by_arm": fire_fraction_by_arm,
        "attr_mass_excess_by_arm": mass_excess_by_arm,
        "jaccard_context_gap_by_arm": jaccard_gap_by_arm,
        "head_n_updates_by_arm": {
            r["arm"]: int(r.get("head_n_updates", 0)) for r in rows},
        "head_norm_entropy_by_arm": {
            r["arm"]: float(r.get("head_norm_entropy", 0.0)) for r in rows},
        "head_max_weight_at_measure_by_arm": {
            r["arm"]: float(r.get("head_max_weight_at_measure", 0.0)) for r in rows},
    }
    print("[smoke] engagement: " + json.dumps(engagement), flush=True)
    if args.dry_run:
        tier1 = {
            "remap_fires_in_fixed_arm",
            "remap_silent_in_control_arm",
            "store_differentiated_at_window_start",
            "episode_restore_effective",
            "pe_varies",
            "bla_ticked",
            "dv_varies_across_arms",
            "fixed_arm_built_no_head",
        }
        failed = [k for k in tier1
                  if isinstance(engagement.get(k), bool) and not engagement[k]]
        if failed:
            raise AssertionError(
                "dry-run engagement check failed: " + ", ".join(failed)
                + " -- the decisive readout is not properly exercised; fix the driver "
                  "before queuing the full grid"
            )
    else:
        full_failures = []
        if not engagement["trainable_heads_trained_past_warmup"]:
            full_failures.append(
                "trainable_heads_trained_past_warmup is False -- at least one "
                f"entropy-weight arm finished with <= {ATTR_HEAD_WARMUP_STEPS} head "
                "updates, so its attribute() was still falling back to the LEGACY proxy "
                "and the arm is a duplicate of ARM_HEAD_FIXED, not a test of the swept "
                "weight"
            )
        if not engagement["attr_mass_excess_varies_across_ew_arms"]:
            full_failures.append(
                "attr_mass_excess_varies_across_ew_arms is False -- the C1 statistic is "
                "bit-identical across every entropy_weight value, so the manipulation "
                "was absorbed rather than changing what the gate selects; a saturation "
                "fingerprint, not a null result"
            )
        if not engagement["baseline_store_matched_across_arms"]:
            full_failures.append(
                "baseline_store_matched_across_arms is False -- the post-P0 baseline "
                "store differs across arms at the same seed, so a head's optimiser "
                "perturbed the encoder trajectory and the arms are not measuring the "
                "same substrate (the detached-input guarantee is broken)"
            )
        if full_failures:
            raise AssertionError(
                "full-run engagement failed: " + " | ".join(full_failures))

    aggregate = aggregate_arm_gates([r["arm_gate"] for r in rows])
    criteria_by_arm = {
        **{arm_id: [
            "C1_attribution_selectivity",
            "C2_context_differentiated_addressing",
            "C4_pe_spike_sparsity",
        ] for arm_id in ON_ARM_IDS},
        ARM_OFF: ["C3_partial_not_wholesale"],
    }
    non_degenerate_by_criterion = arm_criteria_non_degenerate(criteria_by_arm, aggregate)

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
        "entropy_weight_sweep": ENTROPY_WEIGHT_SWEEP,
        "p0_episodes": P0_EPISODES if not args.dry_run else 3,
        "measure_episodes": (
            MEASURE_EPISODES if not args.dry_run else 2 * MIN_WINDOWS_PER_CONTEXT),
        "steps_per_episode": STEPS_PER_EPISODE if not args.dry_run else 15,
        "episodes_per_run": (
            TOTAL_EPISODES if not args.dry_run
            else 3 + 2 * MIN_WINDOWS_PER_CONTEXT),
        "measurement_baseline_restore": (
            "ContextMemory restored to the end-of-P0 snapshot at the start of EVERY "
            "measurement episode, identically in ALL arms"
        ),
        "remap_sigma_on_all_on_arms": REMAP_SIGMA_ON,
        "attribution_head_arms": {a["arm_id"]: a["head"] for a in ARMS},
        "entropy_weight_by_arm": EW_BY_ARM,
        "attr_head_hyperparameters_shared": {
            "warmup_steps": ATTR_HEAD_WARMUP_STEPS,
            "lr": ATTR_HEAD_LR,
            "key_dim": ATTR_HEAD_KEY_DIM,
            "temperature_init": ATTR_HEAD_TEMPERATURE_INIT,
        },
        "supersedes": SUPERSEDES,
        "prior_runs_not_superseded": PRIOR_RUNS,
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
        "remap_sigma_p0_all_arms": P0_REMAP_SIGMA,
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
        "supersedes": SUPERSEDES,
        "prior_runs_not_superseded": PRIOR_RUNS,
        "dispatch_mode": "targeted_probe",
        "seed_policy": "distinct_seeds",
        "experiment_purpose_note": (
            "CONFIG/HYPERPARAMETER RETUNE of MECH-074d, routed from the confirmed "
            "autopsy failure_autopsy_V3-EXQ-906a_894b_2026-08-09 (Target 2), which the "
            "2026-08-10 governance cycle used to amend substrate_queue.json's SD-035 "
            "entry with a narrower failure_record target. V3-EXQ-894/894a/894b all "
            "FAIL/weakens; 894a closed the PE-sigma dimension (Spearman -1.0), 894b "
            "closed the training-budget dimension (training_budget_helps measured "
            "False) and showed the trainable head recovers C1 fully at the module's "
            "default entropy_weight=0.02 but leaves C2 flat -- exactly the failure mode "
            "the module's own docstring (design decision 3) predicted for an entropy_ "
            "weight too large relative to the MSE term. This run holds every other "
            "dimension fixed at 894b's values and sweeps ONLY entropy_weight over "
            f"{ENTROPY_WEIGHT_SWEEP}, on a single P0-only training budget. It does NOT "
            "supersede 894/894a/894b -- those remain valid evidence about the fixed rule "
            "and the default-weight trainable head, and are exactly what motivated this "
            "retune. Wall-independent: every readout is internal to the ContextMemory "
            "store the remap writes to and the BLA gate itself; no behavioural outcome "
            "is read, so no downstream wall can gate the result."
        ),
        "acceptance": ev,
        "arm_results": rows,
        "per_seed_results": ev["per_seed"],
        "per_arm_results": ev["per_arm"],
        "head_contrast": ev["head_contrast"],
        "entropy_weight_sweep": ENTROPY_WEIGHT_SWEEP,
        "dose_response": {
            "spearman_mass_excess_vs_entropy_weight": (
                ev["spearman_mass_excess_vs_entropy_weight"]),
            "spearman_jaccard_gap_vs_entropy_weight": (
                ev["spearman_jaccard_gap_vs_entropy_weight"]),
            "matches_docstring_predicted_direction": (
                ev["dose_response_matches_docstring_direction"]),
            "predicted_direction": (
                "mass_excess (C1) should rise with entropy_weight (POSITIVE rho); "
                "jaccard_gap (C2) should fall with entropy_weight (NEGATIVE rho) -- "
                "per ree_core/amygdala/attribution_head.py design decision (3)"
            ),
        },
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
                  "no ON arm had a green cell: the remap mechanism was not exercised "
                  "well enough under any entropy_weight value to judge MECH-074d; "
                  "self-routed substrate_not_ready_requeue")
        ),
        "summary": (
            f"{QUEUE_ID} MECH-074d entropy_weight/MSE balance retune "
            f"(sweep={ENTROPY_WEIGHT_SWEEP}, sigma held at {REMAP_SIGMA_ON}): "
            f"outcome={ev['outcome']} direction={direction} "
            f"passing_arms={ev['passing_arm_ids']} best_arm={ev['best_arm_id']} "
            f"best_entropy_weight={ev['best_entropy_weight']} "
            f"attribution_recovers={ev['attribution_recovers']} "
            f"fixed_replicates_894={ev['fixed_replicates_894']} "
            f"dose_response_matches_docstring="
            f"{ev['dose_response_matches_docstring_direction']} "
            f"spearman_mass_vs_ew={ev['spearman_mass_excess_vs_entropy_weight']} "
            f"spearman_jaccard_vs_ew={ev['spearman_jaccard_gap_vs_entropy_weight']} "
            f"(any-arm C1={ev['c1_attribution_selectivity_met']} "
            f"C2={ev['c2_context_differentiated_addressing_met']} "
            f"C3={ev['c3_partial_not_wholesale_met']} "
            f"C4={ev['c4_pe_spike_sparsity_met']}) "
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
