"""V3-EXQ-947 -- MECH-314b live per-candidate path: 2x2 validation
(314b ON/OFF  x  candidate-pool first-action diversity floor default/raised).

WHAT THIS TESTS. MECH-314b (frontopolar uncertainty-driven curiosity) became
per-candidate-CAPABLE on 2026-08-08 (the ARC-065 GAP-A Phase-2 slot:
agent._curiosity_per_candidate_uncertainty feeds
StructuredCuriosity.compute_score_bias a genuine [K] vector) and genuinely LIVE
on 2026-08-22 (ree-v3 88287f11c6, the SD-063 E2WorldUncertaintyHead online
training loop). MECH-314b carries pending_retest_after_substrate: true,
reinstated by /governance 2026-07-20 after failure_autopsy_V3-EXQ-604c. This is
that retest.

WHY 2x2 AND NOT A SINGLE-FACTOR ON/OFF. The SD-063 keystone
(sd063_online_head_training_keystone_2026-08-22.md sec 5) found the candidate
pool, not the head, is the binding constraint: only ~2.0-2.4 distinct first
actions across K=32 candidates, so a head with ~10x per-action differentiation
can express at most a 2-valued vector. The follow-on spike
(mech314b_first_action_diversity_spike_2026-08-23.md) showed that ceiling is one
config integer -- support_preserving_min_first_action_classes, default 2,
hardcoded to 2 at all 261 driver call sites -- and that raising it to action_dim
lifts measured diversity to ~4.98/5 through the production SP-CEM path with no
architecture change. So the diversity floor is crossed IN as a factor: without
it, a null would be unreadable (proposer starvation vs. a genuine 314b null).

PRE-RUN MEASUREMENT THAT MOTIVATES THE DESIGN (this session's 2.5a probe, real
CausalGridWorldV2 rollouts, seed 71, 2 ep x 30 steps, trained head):

    arm                    mean first-action classes   committed-action diffs vs OFF
    314b ON, floor 2                 2.167                        0 / 60
    314b ON, floor 5 (=action_dim)   4.983                       15 / 60

At the DEFAULT floor 314b is behaviourally inert; at the raised floor it is not.
A single-factor ON/OFF experiment at the default floor would have recorded a
false null against MECH-314b. That is the interaction this experiment measures
at proper seed count and training budget.

DV-SYMMETRY DECLARATION (mandatory; the V3-EXQ-604c failure class).
604c's DVs were invariant under its own manipulation BY ARITHMETIC: 314b was a
Phase-1 broadcast scalar -- one constant added uniformly across all K candidates
-- and a uniform additive constant cannot move an argmax or survive softmax
normalisation, so its selection-level delta was exactly 0.0 before the run
started. Per arm here:

  * ARM_B314_ON_* : DV is argmax over candidate total score. Its symmetry group
    is (uniform additive constants) x (monotone rescalings). The manipulation
    supplies a per-candidate vector whose CROSS-CANDIDATE relative spread is
    MEASURED non-zero (probe: 0.0365 at floor 2, 0.1948 at floor 5) and is
    asserted at runtime by the pcv_carries_cross_candidate_span precondition.
    A non-constant vector is not a uniform additive constant, so the
    manipulation is NOT invariant under the DV's symmetry group. This is the
    exact property 604c lacked, and it is measured, never assumed.
  * ARM_B314_OFF_*: the 314b contribution is a uniform broadcast, so its
    curiosity deviation IS annihilated by the argmax -- by construction. That is
    the point: these arms are the paired reference, and their flatness is
    asserted (structurally_flat_curiosity_deviation), not discovered.
  * FLOOR factor: manipulates the candidate SET's first-action support. The DV
    is not a set-symmetric aggregate (it is an argmax over a changed support),
    so it is not invariant under candidate permutation either.

READINESS-GATE CORRECTION HONOURED (staged doc sec 5, corrected 2026-08-23).
`last_uncertainty_dev_range > 0` is NECESSARY BUT NOT SUFFICIENT -- an UNTRAINED
head passes it on 320/320 ticks with a LARGER absolute range than a trained one.
The discriminator is RELATIVE spread. This driver therefore gates on a 5-CLASS
head probe (predictive_variance evaluated on all action_dim one-hots at one
z_world), NOT on the latched get_state() value: the latched value's last batch
is the CANDIDATE pool, whose composition is itself the floor manipulation, so
reading it would make the readiness statistic arm-dependent and non-comparable
to the keystone's untrained 0.14-0.26 / trained 1.81-2.37 bands. The >= 1.0
threshold is PROPOSED AND UNVALIDATED (keystone sec 4, 3 seeds, one env); it is
carried here as a readiness precondition and its measured value is recorded on
every cell so this run contributes the independent re-measurement that section
asked for.

TWO DIVERGENCE DVs, and why both.
  * YOKED (primary, load-bearing, NON-COMPOUNDING): the ON agent is stepped on
    the OFF agent's trajectory -- both see an identical observation sequence --
    and divergence is counted per tick as "ON's committed action != OFF's".
    Every tick is a paired comparison at an identical world state, so this
    measures argmin-relevant influence with no trajectory compounding.
  * FREE-RUNNING (secondary, behavioural): the ON agent runs its own
    trajectory. After the first divergence the two histories differ, so this is
    a behavioural SENSITIVITY measure, not a per-tick argmin flip rate. It is
    recorded and reported as such, never as the load-bearing statistic.

SLEEP: none (no sleep flag is set; no SLEEP DRIVER line required).

Substrate-defect note (skill step 2.5c): three open `corrupting` substrate_queue
entries list files this driver's agent imports at module level --
mode-governance-engagement (SalienceCoordinator.tick box clamp),
MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (agent.py::run_sws_schema_pass), and
contextmemory-write-path-addressing-degeneracy (e1_deep.py ContextMemory.write).
The first two are not in this driver's causal path: no mode-governance /
salience-coordinator knob is enabled, and no sleep pass is ever run. The third
(ContextMemory.write single-slot fixed point) IS potentially in path via E1, but
is applied IDENTICALLY to every arm and does not interact with the 314b flag or
the diversity floor, so it cannot manufacture a cross-arm difference -- it biases
the load-bearing contrast toward the null, not toward a false positive. Recorded
here and in the queue-entry note so any later autopsy sees it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "evidence"

EXPERIMENT_TYPE = "v3_exq_947_mech314b_percandidate_2x2_diversity_validation"
QUEUE_ID = "V3-EXQ-947"
CLAIM_IDS = ["MECH-314b"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [71, 101, 202, 303, 404]
EPISODES = 4
STEPS_PER_EPISODE = 80
WORLD_DIM = 32
SELF_DIM = 32

E2U_WARMUP_STEPS = 100
E2U_BATCH_SIZE = 32

# Set ONLY by --dry-run, so the smoke can train the head inside its tiny tick
# budget and therefore actually demonstrate the decisive readout can move.
# A real run always uses E2U_WARMUP_STEPS.
_ACTIVE_WARMUP = E2U_WARMUP_STEPS

# --- PRE-REGISTERED THRESHOLDS (fixed here, never derived from this run) ---
# C1: at the RAISED floor, 314b ON must move the committed action away from its
# paired OFF reference on more than this fraction of yoked ticks.
YOKED_DIVERGENCE_FLOOR = 0.02
# C2: the interaction. Raised-floor divergence must exceed default-floor
# divergence by at least this much.
INTERACTION_MARGIN = 0.02
# Readiness: the keystone's PROPOSED (unvalidated) trained/untrained separator.
HEAD_REL_SPREAD_FLOOR = 1.0
# Readiness: the per-candidate vector must carry some cross-candidate span.
PCV_REL_SPREAD_FLOOR = 0.0
# Readiness: the floor manipulation must actually land.
DIVERSITY_TOLERANCE = 0.5

ARM_OFF_DEFAULT = "ARM_B314_OFF_FLOOR_DEFAULT"
ARM_ON_DEFAULT = "ARM_B314_ON_FLOOR_DEFAULT"
ARM_OFF_RAISED = "ARM_B314_OFF_FLOOR_RAISED"
ARM_ON_RAISED = "ARM_B314_ON_FLOOR_RAISED"


def _arm_id(b314_on: bool, raised: bool) -> str:
    if b314_on:
        return ARM_ON_RAISED if raised else ARM_ON_DEFAULT
    return ARM_OFF_RAISED if raised else ARM_OFF_DEFAULT


def build_config(b314_on: bool, floor: int) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        support_preserving_min_first_action_classes=floor,
    )
    cfg.use_structured_curiosity = True
    # THE 314b FACTOR. "broadcast" is the Phase-1 uniform scalar (argmin-inert
    # by construction); "e2_predictive_variance" is the live per-candidate path.
    cfg.curiosity_uncertainty_source = (
        "e2_predictive_variance" if b314_on else "broadcast"
    )
    # The SD-063 head is instantiated AND trained on every arm, so the two 314b
    # levels differ only in whether its output is CONSUMED per-candidate. This
    # keeps the head's training trajectory a controlled, not a confounded,
    # variable.
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = True
    cfg.latent.e2_world_uncertainty_warmup_steps = _ACTIVE_WARMUP
    cfg.latent.e2_world_uncertainty_batch_size = E2U_BATCH_SIZE
    return cfg


def config_slice(b314_on: bool, floor: int, action_dim: int) -> Dict[str, Any]:
    """Exactly what this cell's computation reads -- no acceptance thresholds."""
    return {
        "env": "CausalGridWorldV2",
        "action_dim": action_dim,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "episodes": EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "use_structured_curiosity": True,
        "curiosity_uncertainty_source": (
            "e2_predictive_variance" if b314_on else "broadcast"
        ),
        "support_preserving_min_first_action_classes": floor,
        "use_e2_world_uncertainty": True,
        "use_e2_world_uncertainty_online_training": True,
        "e2_world_uncertainty_warmup_steps": _ACTIVE_WARMUP,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
    }


def head_5class_relative_spread(agent: REEAgent) -> Optional[float]:
    """Arm-INDEPENDENT readiness statistic: evaluate the head on every action
    class at one z_world and return (max-min)/mean.

    Deliberately NOT get_state()'s latched
    e2_world_uncertainty_last_pvar_relative_spread -- that value's batch is
    whatever was last passed in, which on an ON arm is the CANDIDATE pool, whose
    composition is the floor manipulation itself. Reading it would make the
    readiness gate arm-dependent and incomparable to the keystone's
    untrained 0.14-0.26 / trained 1.81-2.37 bands.
    """
    head = getattr(agent, "e2_world_uncertainty", None)
    if head is None or agent._current_latent is None:
        return None
    z0 = agent._current_latent.z_world.detach()
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    z0 = z0[:1]
    adim = int(agent.config.e2.action_dim)
    acts = torch.eye(adim, device=z0.device, dtype=z0.dtype)
    z0_k = z0.expand(adim, -1)
    with torch.no_grad():
        pvar = head.predictive_variance(z0_k, acts).reshape(-1)
    mean = float(pvar.mean())
    if mean == 0.0:
        return 0.0
    return float((pvar.max() - pvar.min()) / abs(mean))


class _Runner:
    """One agent, driven tick by tick, recording the per-tick observables.

    OWNS ITS OWN RNG STREAM. Two agents stepped alternately from one global
    torch stream interleave their draws, so the second never sees the state the
    first saw -- and since E3 selection samples via torch.multinomial, two
    agents with IDENTICAL config then diverge for purely stream-positional
    reasons. Measured before this fix: an OFF-vs-OFF negative control diverged
    19/80 (floor 2) and 37/80 (floor 5), i.e. the yoked DV was reading RNG
    interleaving rather than the 314b factor. Each runner therefore snapshots
    and restores its own generator state around every tick, so a paired ON/OFF
    comparison differs in the manipulation and nothing else.
    """

    def __init__(self, cfg: REEConfig) -> None:
        self.agent = REEAgent(cfg)
        # Captured AFTER construction, with the caller's reset_all_rng(seed)
        # still in force, so two identically-configured runners start from an
        # identical stream position.
        self._rng_state = torch.get_rng_state()
        self.world_dim = cfg.latent.world_dim
        self.n_ticks = 0
        self.class_sum = 0
        self.pcv_spreads: List[float] = []
        self.head_spreads: List[float] = []
        self.unc_dev_ranges: List[float] = []
        self.bias_ranges: List[float] = []
        self.clamp_fracs: List[float] = []

    def reset_episode(self) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.reset()
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def choose(self, obs: Dict[str, Any]) -> int:
        """Advance one tick and return the committed action index.

        Runs entirely on THIS runner's private RNG stream (see class docstring).
        """
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            return self._choose_inner(obs)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def _choose_inner(self, obs: Dict[str, Any]) -> int:
        agent = self.agent
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick")
            else torch.zeros(1, self.world_dim, device=agent.device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        if candidates:
            self.n_ticks += 1
            classes = {
                agent.hippocampal._trajectory_first_action_class(c)
                for c in candidates
            }
            self.class_sum += len(classes)
            vec = agent._curiosity_per_candidate_uncertainty(candidates)
            if vec is not None and vec.numel() > 1:
                mean = float(vec.abs().mean())
                if mean > 0.0:
                    self.pcv_spreads.append(
                        float((vec.max() - vec.min()) / mean)
                    )
            spread = head_5class_relative_spread(agent)
            if spread is not None:
                self.head_spreads.append(spread)
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]),
        )
        action = agent.select_action(candidates, ticks)
        state = agent.curiosity.get_state()
        self.unc_dev_ranges.append(float(state["last_uncertainty_dev_range"]))
        self.bias_ranges.append(float(state["last_bias_range"]))
        self.clamp_fracs.append(float(state["last_clamp_saturated_frac"]))
        return int(action.argmax(dim=-1).item())

    def observe(self, harm: Any) -> None:
        ambient = torch.get_rng_state()
        torch.set_rng_state(self._rng_state)
        try:
            self.agent.update_residue(harm)
        finally:
            self._rng_state = torch.get_rng_state()
            torch.set_rng_state(ambient)

    def summary(self) -> Dict[str, Any]:
        def _mean(xs: List[float]) -> float:
            return (sum(xs) / len(xs)) if xs else 0.0

        head = self.agent.e2_world_uncertainty
        hstate = head.get_state() if head is not None else {}
        return {
            "n_candidate_ticks": self.n_ticks,
            "mean_distinct_first_action_classes": (
                self.class_sum / self.n_ticks if self.n_ticks else 0.0
            ),
            "pcv_relative_spread_mean": _mean(self.pcv_spreads),
            "n_pcv_nonnull_ticks": len(self.pcv_spreads),
            "head_5class_relative_spread_mean": _mean(self.head_spreads),
            "head_5class_relative_spread_final": (
                self.head_spreads[-1] if self.head_spreads else 0.0
            ),
            "last_uncertainty_dev_range_mean": _mean(self.unc_dev_ranges),
            "last_bias_range_mean": _mean(self.bias_ranges),
            "last_clamp_saturated_frac_mean": _mean(self.clamp_fracs),
            "head_n_train_steps": int(
                hstate.get("e2_world_uncertainty_n_train_steps", 0)
            ),
            "head_latched_pvar_relative_spread": float(
                hstate.get("e2_world_uncertainty_last_pvar_relative_spread", 0.0)
            ),
        }


def run_yoked_pair(seed: int, floor: int, episodes: int, steps: int,
                   zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """OFF agent drives the environment; ON agent is stepped on the SAME
    observation sequence and its committed action compared per tick.

    Non-compounding by construction: both agents see an identical world state at
    every tick, so a divergence is an argmin flip, not an accumulated history
    difference.
    """
    # CRITICAL: reset the global RNG before EACH agent construction. Building
    # them back to back draws weight init from one continuing stream, so the two
    # agents would differ in every weight and the divergence would measure
    # initialisation, not the 314b factor. (Caught by the dry-run smoke: yoked
    # divergence read 1.0 at BOTH floors before this reset was added.)
    reset_all_rng(seed)
    off = _Runner(build_config(b314_on=False, floor=floor))
    reset_all_rng(seed)
    on = _Runner(build_config(b314_on=True, floor=floor))
    reset_all_rng(seed)
    env = CausalGridWorldV2()
    n_cmp = 0
    n_diff = 0
    per_episode: List[Dict[str, Any]] = []
    for ep in range(episodes):
        _, obs = env.reset()
        off.reset_episode()
        on.reset_episode()
        ep_cmp = 0
        ep_diff = 0
        for _ in range(steps):
            a_off = off.choose(obs)
            a_on = on.choose(obs)
            n_cmp += 1
            ep_cmp += 1
            if a_off != a_on:
                n_diff += 1
                ep_diff += 1
            _f, harm, _d, _i, obs = env.step(a_off)
            off.observe(harm)
            on.observe(harm)
        per_episode.append(
            {"episode": ep, "n_compared": ep_cmp, "n_diverged": ep_diff,
             "divergence_frac": ep_diff / ep_cmp if ep_cmp else 0.0}
        )
        print(f"  [train] yoked seed={seed} floor={floor} ep {ep + 1}/{episodes} "
              f"diverged={ep_diff}/{ep_cmp}", flush=True)
    zg.observe(off.agent)
    zg.observe(on.agent)
    return {
        "off_summary": off.summary(),
        "on_summary": on.summary(),
        "yoked_n_compared": n_cmp,
        "yoked_n_diverged": n_diff,
        "yoked_divergence_frac": n_diff / n_cmp if n_cmp else 0.0,
        "yoked_per_episode": per_episode,
    }


def run_freerunning(seed: int, floor: int, b314_on: bool, episodes: int,
                    steps: int, zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """One agent on its own trajectory. Secondary, behavioural."""
    # Same pairing discipline as run_yoked_pair: the ON and OFF free-running
    # cells must start from identical weights for their action sequences to be
    # comparable at all.
    reset_all_rng(seed)
    runner = _Runner(build_config(b314_on=b314_on, floor=floor))
    reset_all_rng(seed)
    env = CausalGridWorldV2()
    actions: List[int] = []
    for ep in range(episodes):
        _, obs = env.reset()
        runner.reset_episode()
        for _ in range(steps):
            a = runner.choose(obs)
            actions.append(a)
            _f, harm, _d, _i, obs = env.step(a)
            runner.observe(harm)
        print(f"  [train] freerun seed={seed} floor={floor} "
              f"b314={'ON' if b314_on else 'OFF'} ep {ep + 1}/{episodes}",
              flush=True)
    zg.observe(runner.agent)
    out = runner.summary()
    out["actions"] = actions
    return out


def paired_control_divergence(seed: int, floor: int, b314_on: bool,
                             episodes: int = 1, steps: int = 20) -> float:
    """INSTRUMENT CONTROL: yoke an arm against ITSELF and return the divergence
    fraction, which MUST be 0.0.

    Two identically-configured agents can only diverge if the pairing is broken.
    This exact control caught the defect that made the first draft of this
    driver meaningless: sharing one global torch RNG stream between the two
    yoked agents produced 19/80 (floor 2) and 37/80 (floor 5) OFF-vs-OFF
    divergence, so the DV was reading stream interleaving rather than the 314b
    factor. It is measured in-run, per floor and per 314b level, so a future
    regression in the per-runner RNG isolation fails the readiness gate instead
    of silently returning a confident number.
    """
    reset_all_rng(seed)
    a = _Runner(build_config(b314_on=b314_on, floor=floor))
    reset_all_rng(seed)
    b = _Runner(build_config(b314_on=b314_on, floor=floor))
    reset_all_rng(seed)
    env = CausalGridWorldV2()
    n = d = 0
    for _ in range(episodes):
        _, obs = env.reset()
        a.reset_episode()
        b.reset_episode()
        for _ in range(steps):
            aa = a.choose(obs)
            bb = b.choose(obs)
            n += 1
            if aa != bb:
                d += 1
            _f, harm, _dn, _i, obs = env.step(aa)
            a.observe(harm)
            b.observe(harm)
    return (d / n) if n else 0.0


PRECONDITIONS = [
    PreconditionSpec(
        name="candidate_pool_diversity_matches_floor",
        description=(
            "mean distinct first-action classes across candidates reaches the "
            "arm's declared support_preserving_min_first_action_classes floor "
            "(the floor-factor manipulation check)"
        ),
        control="every tick's own candidate pool on a real CausalGridWorldV2 rollout",
        threshold=0.0,  # set per-arm via met_overrides; see _arm_specs
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="paired_control_is_bit_identical",
        description=(
            "INSTRUMENT CONTROL: an arm yoked against ITSELF diverges on 0 "
            "ticks. Non-zero means the two yoked agents differ in something "
            "other than the manipulation and the whole DV is void"
        ),
        control="same arm vs same arm, identical seed and config",
        threshold=1e-9,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="pcv_carries_cross_candidate_span",
        description=(
            "the per-candidate uncertainty vector's cross-candidate relative "
            "spread is non-zero -- i.e. the 314b manipulation is NOT a uniform "
            "additive constant and is therefore not annihilated by the argmax "
            "(the V3-EXQ-604c DV-symmetry failure)"
        ),
        control="live candidate pools with a trained head; probe measured 0.0365 (floor 2) / 0.1948 (floor 5)",
        threshold=PCV_REL_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"]),
        applies_note=(
            "OFF arms consume no per-candidate vector by construction "
            "(curiosity_uncertainty_source='broadcast'), so a cross-candidate "
            "span is not the right question there -- their flatness is the "
            "paired reference, not a failure"
        ),
    ),
    PreconditionSpec(
        name="head_5class_relative_spread_trained",
        description=(
            "SD-063 head predictive_variance evaluated on all action classes at "
            "one z_world shows relative spread above the keystone's PROPOSED "
            "(unvalidated) trained/untrained separator -- untrained band "
            "0.14-0.26, trained band 1.81-2.37"
        ),
        control="all action_dim one-hots at a single z_world, arm-independent by construction",
        threshold=HEAD_REL_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"]),
        applies_note=(
            "the head trains on every arm but is CONSUMED only on ON arms; "
            "its readiness gates the 314b channel, not the broadcast reference"
        ),
    ),
    PreconditionSpec(
        name="curiosity_channel_carries_argmin_relevant_span",
        description=(
            "StructuredCuriosity.last_bias_range > 0 -- the surviving "
            "argmin-relevant deviation is non-zero"
        ),
        control="live per-candidate curiosity bias on a trained head",
        threshold=0.0,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"]),
        applies_note=(
            "on OFF arms every enabled sub-flavour is a uniform broadcast, so "
            "last_bias_range is 0.0 BY CONSTRUCTION (measured 0.000e+00 in the "
            "2.5a probe). Asserting it there would make the OFF reference "
            "structurally un-passable and collapse the 2x2 back to one factor"
        ),
    ),
    PreconditionSpec(
        name="curiosity_clamp_not_fully_saturated",
        description=(
            "last_clamp_saturated_frac stays below its (K-1)/K ceiling -- the "
            "ranking is not fully compressed by the shared deviation clamp"
        ),
        control="live per-candidate curiosity bias on a trained head",
        threshold=0.95,
        direction="upper",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"]),
        applies_note="a broadcast-only channel never reaches the clamp rail",
    ),
]


def _arm_specs(floor: int) -> List[PreconditionSpec]:
    """The diversity precondition's threshold is the arm's own floor."""
    specs: List[PreconditionSpec] = []
    for spec in PRECONDITIONS:
        if spec.name == "candidate_pool_diversity_matches_floor":
            specs.append(
                PreconditionSpec(
                    name=spec.name,
                    description=spec.description,
                    control=spec.control,
                    threshold=float(floor) - DIVERSITY_TOLERANCE,
                    direction="lower",
                    kind="readiness",
                    structural_max=lambda ctx: float(ctx["action_dim"]),
                )
            )
        else:
            specs.append(spec)
    return specs


def _arm_ctx(b314_on: bool, floor: int, action_dim: int) -> Dict[str, Any]:
    return {
        "arm_id": _arm_id(b314_on, floor != 2),
        "b314_on": b314_on,
        "floor": floor,
        "action_dim": action_dim,
    }


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    probe_env = CausalGridWorldV2()
    action_dim = int(probe_env.action_dim)
    floors = [2, action_dim]
    zg = ZGoalStreamAccumulator()

    # Design-time refusal BEFORE compute: prove no applicable precondition is
    # unsatisfiable from the pre-registered config.
    for floor in floors:
        for b314_on in (False, True):
            assert_no_structurally_unsatisfiable_gate(
                _arm_specs(floor),
                [_arm_ctx(b314_on, floor, action_dim)],
            )

    # INSTRUMENT CONTROL, before any scored compute: an arm yoked against
    # itself must diverge on zero ticks.
    control_div: Dict[str, float] = {}
    for floor in floors:
        for b314_on in (False, True):
            aid = _arm_id(b314_on, floor != 2)
            control_div[aid] = paired_control_divergence(
                seeds[0], floor, b314_on,
                episodes=1, steps=(6 if dry_run else 20))
            print(f"[control] {aid}: self-yoked divergence = "
                  f"{control_div[aid]:.6f} (must be 0)", flush=True)

    arm_results: List[Dict[str, Any]] = []
    for floor in floors:
        raised = floor != 2
        for seed in seeds:
            print(f"Seed {seed} Condition floor{floor}_yoked", flush=True)
            slice_off = config_slice(False, floor, action_dim)
            with arm_cell(seed, config_slice=slice_off,
                          script_path=Path(__file__),
                          config_slice_declared=True,
                          include_driver_script_in_hash=False) as cell:
                paired = run_yoked_pair(seed, floor, episodes, steps, zg)
                off_row = {
                    "arm_id": _arm_id(False, raised),
                    "seed": seed,
                    "floor": floor,
                    "b314_on": False,
                    **paired["off_summary"],
                }
                cell.stamp(off_row)
            arm_results.append(off_row)
            print(f"verdict: {'PASS' if off_row['n_candidate_ticks'] > 0 else 'FAIL'}",
                  flush=True)

            print(f"Seed {seed} Condition floor{floor}_b314on", flush=True)
            slice_on = config_slice(True, floor, action_dim)
            with arm_cell(seed, config_slice=slice_on,
                          script_path=Path(__file__),
                          config_slice_declared=True,
                          include_driver_script_in_hash=False) as cell:
                free = run_freerunning(seed, floor, True, episodes, steps, zg)
                free_actions = free.pop("actions")
                off_free = run_freerunning(seed, floor, False, episodes, steps, zg)
                off_actions = off_free.pop("actions")
                n = min(len(free_actions), len(off_actions))
                n_diff = sum(1 for i in range(n)
                             if free_actions[i] != off_actions[i])
                on_row = {
                    "arm_id": _arm_id(True, raised),
                    "seed": seed,
                    "floor": floor,
                    "b314_on": True,
                    **paired["on_summary"],
                    "yoked_n_compared": paired["yoked_n_compared"],
                    "yoked_n_diverged": paired["yoked_n_diverged"],
                    "yoked_divergence_frac": paired["yoked_divergence_frac"],
                    "yoked_per_episode": paired["yoked_per_episode"],
                    "freerun_n_compared": n,
                    "freerun_n_diverged": n_diff,
                    "freerun_divergence_frac": (n_diff / n) if n else 0.0,
                    "freerun_off_reference_summary": off_free,
                }
                cell.stamp(on_row)
            arm_results.append(on_row)
            print(f"verdict: {'PASS' if on_row['n_candidate_ticks'] > 0 else 'FAIL'}",
                  flush=True)

    # ---- per-arm readiness gates (regime-conditioned, never whole-run ANDed) --
    arm_gates = []
    for floor in floors:
        raised = floor != 2
        for b314_on in (False, True):
            aid = _arm_id(b314_on, raised)
            rows = [r for r in arm_results if r["arm_id"] == aid]
            if not rows:
                continue
            ctx = _arm_ctx(b314_on, floor, action_dim)
            measured = {
                "paired_control_is_bit_identical": control_div.get(aid, 1.0),
                # worst cell, never the mean -- the recompute must test the same
                # statistic the `met` quantifier does.
                "candidate_pool_diversity_matches_floor": min(
                    r["mean_distinct_first_action_classes"] for r in rows),
                "pcv_carries_cross_candidate_span": min(
                    r["pcv_relative_spread_mean"] for r in rows),
                "head_5class_relative_spread_trained": min(
                    r["head_5class_relative_spread_final"] for r in rows),
                "curiosity_channel_carries_argmin_relevant_span": min(
                    r["last_bias_range_mean"] for r in rows),
                "curiosity_clamp_not_fully_saturated": max(
                    r["last_clamp_saturated_frac_mean"] for r in rows),
            }
            arm_gates.append(
                evaluate_arm_gate(aid, ctx, _arm_specs(floor), measured)
            )
    aggregate = aggregate_arm_gates(arm_gates)

    # ---- pre-registered criteria ------------------------------------------
    def _cells(aid: str) -> List[Dict[str, Any]]:
        return [r for r in arm_results if r["arm_id"] == aid]

    def _mean_of(aid: str, key: str) -> float:
        rows = _cells(aid)
        return (sum(r[key] for r in rows) / len(rows)) if rows else 0.0

    yoked_default = _mean_of(ARM_ON_DEFAULT, "yoked_divergence_frac")
    yoked_raised = _mean_of(ARM_ON_RAISED, "yoked_divergence_frac")
    pcv_default = _mean_of(ARM_ON_DEFAULT, "pcv_relative_spread_mean")
    pcv_raised = _mean_of(ARM_ON_RAISED, "pcv_relative_spread_mean")

    c1 = yoked_raised > YOKED_DIVERGENCE_FLOOR
    c2 = (yoked_raised - yoked_default) >= INTERACTION_MARGIN
    c3 = pcv_raised > pcv_default

    green = set(aggregate["green_arms"])
    overall_pass = bool(c1 and c2 and (ARM_ON_RAISED in green))

    criteria = [
        {"name": "C1_b314_moves_committed_action_at_raised_floor",
         "load_bearing": True, "passed": bool(c1),
         "measured": yoked_raised, "threshold": YOKED_DIVERGENCE_FLOOR},
        {"name": "C2_effect_scales_with_diversity_floor",
         "load_bearing": True, "passed": bool(c2),
         "measured": yoked_raised - yoked_default,
         "threshold": INTERACTION_MARGIN},
        {"name": "C3_pcv_span_scales_with_diversity_floor",
         "load_bearing": False, "passed": bool(c3),
         "measured": pcv_raised - pcv_default, "threshold": 0.0},
    ]
    combination_rule = (
        "overall_pass = C1 AND C2 AND (ARM_B314_ON_FLOOR_RAISED readiness gate "
        "green). C3 is supporting mechanism evidence and does NOT gate the "
        "outcome. A FAIL with a GREEN raised-floor gate is a genuine negative "
        "for MECH-314b (the channel carries no argmin-relevant influence even "
        "with a saturated candidate pool); a FAIL with a RED gate is "
        "substrate-not-ready and carries no claim evidence."
    )

    if not overall_pass and ARM_ON_RAISED not in green:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        direction = "supports"
        label = "b314_percandidate_path_carries_selection_authority_at_raised_diversity"
    elif c1 and not c2:
        direction = "mixed"
        label = "b314_effect_present_but_not_floor_dependent"
    else:
        direction = "weakens"
        label = "b314_percandidate_path_inert_at_selection_despite_saturated_pool"

    criteria_nd = arm_criteria_non_degenerate(
        {
            ARM_ON_RAISED: ["C1_b314_moves_committed_action_at_raised_floor"],
            ARM_ON_DEFAULT: ["C3_pcv_span_scales_with_diversity_floor"],
        },
        aggregate,
        extra={
            "C2_effect_scales_with_diversity_floor": bool(
                ARM_ON_RAISED in green and ARM_ON_DEFAULT in green
            ),
        },
    )
    criteria_nd["C2_effect_scales_with_diversity_floor"] = bool(
        ARM_ON_RAISED in green and ARM_ON_DEFAULT in green
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": "PASS" if overall_pass else "FAIL",
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "dry_run": dry_run,
        "metrics": {
            "yoked_divergence_frac_floor_default": yoked_default,
            "yoked_divergence_frac_floor_raised": yoked_raised,
            "yoked_divergence_interaction": yoked_raised - yoked_default,
            "pcv_relative_spread_floor_default": pcv_default,
            "pcv_relative_spread_floor_raised": pcv_raised,
            "freerun_divergence_frac_floor_default": _mean_of(
                ARM_ON_DEFAULT, "freerun_divergence_frac"),
            "freerun_divergence_frac_floor_raised": _mean_of(
                ARM_ON_RAISED, "freerun_divergence_frac"),
            "mean_distinct_first_action_classes_floor_default_on": _mean_of(
                ARM_ON_DEFAULT, "mean_distinct_first_action_classes"),
            "mean_distinct_first_action_classes_floor_raised_on": _mean_of(
                ARM_ON_RAISED, "mean_distinct_first_action_classes"),
            "head_5class_relative_spread_floor_default_on": _mean_of(
                ARM_ON_DEFAULT, "head_5class_relative_spread_final"),
            "head_5class_relative_spread_floor_raised_on": _mean_of(
                ARM_ON_RAISED, "head_5class_relative_spread_final"),
            "paired_control_divergence_max": max(control_div.values())
            if control_div else 1.0,
        },
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": arm_results,
        "per_arm_gate": aggregate["per_arm_gate"],
        "non_degenerate": aggregate["non_degenerate"],
        "degeneracy_reason": aggregate["degeneracy_reason"],
        "interpretation": {
            "label": label,
            "preconditions": aggregate["adjudication_preconditions"],
            "preconditions_scope_note": aggregate.get(
                "preconditions_scope_note", ""),
            "criteria_non_degenerate": criteria_nd,
        },
        "dv_symmetry_declaration": {
            "ARM_B314_ON_*": (
                "DV is argmax over candidate total score; symmetry group is "
                "uniform additive constants x monotone rescalings. The "
                "manipulation supplies a NON-constant per-candidate vector "
                "(pcv_relative_spread asserted > 0 at runtime), so it is not "
                "invariant under that group. Measured, not assumed."
            ),
            "ARM_B314_OFF_*": (
                "the 314b contribution is a uniform broadcast and IS "
                "annihilated by the argmax by construction -- this is the "
                "paired reference, declared, not discovered (the V3-EXQ-604c "
                "class made explicit)."
            ),
            "FLOOR_FACTOR": (
                "manipulates the candidate set's first-action support; the DV "
                "is an argmax over a changed support, not a set-symmetric "
                "aggregate, so it is not invariant under candidate permutation."
            ),
        },
        "custom_information": {
            "readiness_statistic_note": (
                "head_5class_relative_spread is computed on all action_dim "
                "one-hots at a single z_world, NOT read from get_state()'s "
                "latched last_pvar_relative_spread, whose batch on an ON arm is "
                "the candidate pool -- i.e. the floor manipulation itself."
            ),
            "head_rel_spread_threshold_status": (
                "HEAD_REL_SPREAD_FLOOR=1.0 is the keystone's PROPOSED and "
                "UNVALIDATED separator (3 seeds, one env). This run records its "
                "measured value on every cell as the independent "
                "re-measurement sd063_online_head_training_keystone sec 4 asked "
                "for; it is not treated here as a settled gate value."
            ),
            "freerun_dv_caveat": (
                "freerun_divergence_frac compounds after the first divergence "
                "and is a behavioural sensitivity measure, NOT a per-tick "
                "argmin flip rate. The load-bearing statistic is the yoked one."
            ),
            "substrate_defect_note": (
                "open corrupting substrate_queue entry "
                "contextmemory-write-path-addressing-degeneracy "
                "(ree_core/predictors/e1_deep.py) is in path via E1 but applies "
                "identically to every arm and does not interact with either "
                "factor, so it biases the load-bearing contrast toward the "
                "null, not toward a false positive."
            ),
        },
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
    }

    full_config = {
        "seeds": seeds,
        "episodes": episodes,
        "steps_per_episode": steps,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "action_dim": action_dim,
        "floors": floors,
        "e2_world_uncertainty_warmup_steps": E2U_WARMUP_STEPS,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
        "thresholds": {
            "YOKED_DIVERGENCE_FLOOR": YOKED_DIVERGENCE_FLOOR,
            "INTERACTION_MARGIN": INTERACTION_MARGIN,
            "HEAD_REL_SPREAD_FLOOR": HEAD_REL_SPREAD_FLOOR,
            "DIVERSITY_TOLERANCE": DIVERSITY_TOLERANCE,
        },
    }

    out_path = write_flat_manifest(
        manifest,
        dry_run=dry_run,
        config=full_config,
        seeds=seeds,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": manifest["outcome"], "manifest": manifest,
            "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    args = ap.parse_args()

    global _ACTIVE_WARMUP
    if args.dry_run:
        # 2 x 40 = 80 ticks with a 20-transition warmup, so the head genuinely
        # trains and the smoke can assert the decisive readout is able to move
        # (a 6-tick smoke at warmup 100 never trains and proves nothing).
        episodes, steps, seeds = 2, 40, [71]
        _ACTIVE_WARMUP = 20
    else:
        episodes, steps, seeds = args.episodes, args.steps, SEEDS

    result = run_experiment(episodes, steps, seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"manifest: {out_path}", flush=True)
    print(json.dumps(result["manifest"]["metrics"], indent=2), flush=True)

    if args.dry_run:
        m = result["manifest"]["metrics"]
        checks = {
            "floor manipulation lands (raised > default classes)":
                m["mean_distinct_first_action_classes_floor_raised_on"]
                > m["mean_distinct_first_action_classes_floor_default_on"],
            "per-candidate vector carries cross-candidate span (ON, raised)":
                m["pcv_relative_spread_floor_raised"] > 0.0,
            "pcv span scales with the floor":
                m["pcv_relative_spread_floor_raised"]
                > m["pcv_relative_spread_floor_default"],
            "head trained above the untrained band (>0.26)":
                m["head_5class_relative_spread_floor_raised_on"] > 0.26,
            "decisive DV can MOVE at the raised floor (yoked > 0)":
                m["yoked_divergence_frac_floor_raised"] > 0.0,
            "INSTRUMENT CONTROL: self-yoked arms are bit-identical (==0)":
                m["paired_control_divergence_max"] == 0.0,
        }
        print("[smoke] decisive-readout engagement checks:", flush=True)
        for label, ok in checks.items():
            print(f"  [{'OK' if ok else 'XX'}] {label}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(
        outcome=_outcome_raw,
        manifest_path=_out_path,
        dry_run=_dry,
    )
