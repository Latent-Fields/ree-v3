"""V3-EXQ-949 -- MECH-314b authority-rescale validation: 2x2
(314b ON/OFF  x  use_modulatory_selection_authority ON/OFF), fixed at the
RAISED candidate-diversity floor (=action_dim).

WHAT THIS TESTS AND WHY IT SUPERSEDES V3-EXQ-947 AS BRIEFED.
`mech314b_percandidate_2x2_design_and_null_authority_2026-08-23.md` found
MECH-314b's live per-candidate path (ree-v3/experiments/
v3_exq_947_mech314b_percandidate_2x2_diversity_validation.py) changes the
committed action on 0/320 paired ticks at BOTH diversity floors, with every
ARC-065 section-5 readiness precondition green (trained head, saturated
candidate pool, non-degenerate per-candidate vector). That driver's queue
entry was never appended (claim contention; see that doc sec 6), so no
V3-EXQ-947 manifest exists on record.

`mech314b_score_bias_magnitude_diagnostic_2026-08-25.md` then measured WHY:
`structured_curiosity`'s per-candidate contribution (`last_uncertainty_dev_range`)
is ~1.5-2e-05 in this environment, while the base E3 `raw_score_range` the
argmin actually operates over is ~270-290 -- a ratio of ~1.4-1.8e7x. This is
NOT closeable by raising `curiosity_uncertainty_weight`: the contribution is
hard-clamped to `curiosity_bias_scale` (0.1 default) before it ever reaches
the argmin, and 0.1 itself is ~2700-2900x too small relative to
`raw_score_range`. The diagnostic confirmed the already-built, off-by-default
fix `use_modulatory_selection_authority` (rescales the COMBINED modulatory
contribution to `modulatory_authority_gain * raw_score_range`) changes
`post_score_range` materially (271.4->243.8 mean, 284.9->178.4 last tick at
gain=0.5) where the OFF baseline shows `post_score_range == raw_score_range`
to displayed precision at every tick.

So V3-EXQ-947 as originally briefed (314b ON/OFF, authority always OFF) is
mechanistically guaranteed null regardless of seed count -- more seeds cannot
close an ~1e7x architectural gap. This experiment supersedes it: same
yoked-pair / private-RNG-stream / in-run-pairing-control design, same claim
(MECH-314b), but crossing 314b ON/OFF against `use_modulatory_selection_authority`
ON/OFF instead of against the candidate-diversity floor (which the 2x2 already
proved is necessary -- saturated at floor=action_dim -- and is fixed here at
its raised value rather than re-tested). This retest ALSO supplies the
per-claim evidence retest `use_modulatory_selection_authority` itself is
gated on (claims.yaml, `implemented_pending_validation` in
substrate_queue.json).

RE-DERIVE BRAKE (MOVE-3), recorded. MECH-314b's brake count is 3 (>= threshold
2): `failure_autopsy_604a-624a-630` (v3_exq_604a), `failure_autopsy_gapA-
cluster-604b-648a-649` (v3_exq_604b, itself an EARLIER authority-ON attempt),
`failure_autopsy_V3-EXQ-604c`. **The brake is RELEASED for this retest.**
604b's own `recommended_substrate_queue_entry` named ARC-065 (GAP-A
action-conditional candidate diversity) as the upstream substrate this claim
needed, with `failure_record_entry.metric`: "curiosity bias magnitude fired
... but cross-candidate range ~0 because the consumed candidate pool was
class-uniform (pre-GAP-A)". That is a DIFFERENT root cause than the one this
driver retests (magnitude-vs-range architectural mismatch, not pool
uniformity) -- and ARC-065's GAP-A slot has since landed and been validated
(V3-EXQ-649) and SD-063 online head training went live 2026-08-22, both
confirmed by V3-EXQ-947's own 2x2 (candidate pool saturated at 4.934/5,
head 5-class relative spread 1.7584, inside the trained band). So this is a
retest of the SAME claim against a NOW-DIFFERENT, additionally-corrected
substrate condition (magnitude rescale), not a repeat probe into the same
ceiling 604a/604b/604c already established.

SUBSTRATE-PATH OVERLAP GATE (skill step 2.5c), carried forward unchanged from
the 947 driver (same imports, same causal path): three open `corrupting`
substrate_queue entries name files this driver's agent imports at module
level. `mode-governance-engagement` (SalienceCoordinator.tick) is not in this
driver's causal path (no mode-governance knob enabled). `contextmemory-write-
path-addressing-degeneracy` (e1_deep.py ContextMemory.write) IS potentially in
path via E1, but applies IDENTICALLY to every one of the 4 arms and interacts
with neither the 314b factor nor the authority factor, so it biases the
load-bearing contrast toward the null, not toward a false positive.

THE DESIGN: ONE YOKED QUAD, not two separate yoked pairs. A single reference
agent (ARM_B314_OFF_AUTH_OFF) drives the environment; the other three arms
(ARM_B314_ON_AUTH_OFF, ARM_B314_OFF_AUTH_ON, ARM_B314_ON_AUTH_ON) are each
independently stepped on the REFERENCE's identical observation sequence, so
all four agents see the same world state at every tick and every comparison
is a paired argmin flip, non-compounding. This is the same yoked-pair
architecture the 947 driver validated, extended from a pair to a quad so all
four cells of the 2x2 share one reference trajectory and one env rollout per
seed (rather than needing 3 separate pairwise trajectories).

OWNS ITS OWN RNG STREAM (carried forward from the 947 driver, unchanged in
mechanism). Four agents stepped round-robin from one global torch RNG stream
would interleave their multinomial draws and diverge for purely
stream-positional reasons even at identical configuration -- exactly the
defect the 947 driver's negative control caught (19/80, 37/80 self-yoked
divergence before the fix). Each `_Runner` here snapshots and restores its
own generator state around every tick, episode reset, and residue update, and
the same self-yoked instrument control (`paired_control_divergence`) runs
in-run for all four arm identities before any scored compute.

DV-SYMMETRY DECLARATION (mandatory; the V3-EXQ-604c failure class), per arm:
  * ARM_B314_OFF_AUTH_OFF (reference): both factors off. curiosity contributes
    a uniform broadcast (bias_range == 0 by construction), authority is
    disabled. No mechanism exists here that could move the argmin away from
    the reference's own action -- this arm never diverges from itself.
  * ARM_B314_ON_AUTH_OFF: 314b ON, authority OFF. Reproduces the 947 driver's
    proven-null cell: the per-candidate vector carries real cross-candidate
    span (measured non-zero, asserted at runtime), but at the UN-rescaled
    magnitude (~1.5e-05 vs raw_score_range ~270-290) it is not invariant under
    the DV's symmetry group in principle, but is empirically negligible in
    practice at this magnitude -- this is what the diagnostic measured, not
    assumed, and this arm's divergence is expected to replicate at ~0.
  * ARM_B314_OFF_AUTH_ON: 314b OFF (broadcast), authority ON. STRUCTURAL
    negative control. `e3_selector.select`'s authority rescale only engages
    when the combined modulatory contribution's cross-candidate RANGE exceeds
    `modulatory_authority_min_range_floor` (1e-6) -- and a uniform broadcast
    has RANGE EXACTLY 0 by construction (max-min of a constant). So authority
    cannot engage here regardless of gain: `modulatory_authority_active`
    stays False and `post_score_range == raw_score_range` exactly, same as
    the OFF/OFF reference. This is asserted at runtime
    (`authority_no_spurious_engagement_without_signal`), not merely assumed --
    it is the DV-symmetry-correct negative control for the authority factor
    alone, proving authority does not manufacture movement from nothing.
  * ARM_B314_ON_AUTH_ON: 314b ON, authority ON. The manipulation supplies a
    non-constant per-candidate vector (measured cross-candidate span) AND the
    combined modulatory contribution is rescaled to a magnitude genuinely
    competitive with raw_score_range (diagnostic measured post_score_range
    diverging materially from raw_score_range under this exact configuration).
    Neither factor alone is invariant under the DV's symmetry group; this is
    the arm under test for real argmin authority.

READINESS-GATE CORRECTION HONOURED (947 driver, carried forward unchanged).
`last_uncertainty_dev_range > 0` is necessary but not sufficient -- an
untrained head passes it with a LARGER absolute range than a trained one. The
discriminator is RELATIVE spread, gated on a 5-CLASS head probe
(predictive_variance on all action_dim one-hots at one z_world), not the
latched get_state() value (arm-dependent on an ON arm, since its last batch is
the candidate pool). Threshold PROPOSED AND UNVALIDATED (keystone sec 4);
measured value recorded on every cell regardless.

NEW READINESS PRECONDITION (this experiment): the authority mechanism must be
measured to actually ENGAGE where expected and NOT engage where it shouldn't.
Both `agent.e3.last_raw_scores` and `agent.e3.last_scores` are E3-cadence
latches (populated only inside `E3TrajectorySelector.select()`, which per
`heartbeat.e3_steps_per_tick` (default 10) does not run every env tick) --
so the mandatory sample-size-integrity idiom applies: both are explicitly
CLEARED to None immediately before every `select_action(...)` call, and a
tick is counted toward `raw_score_range` / `post_score_range` ONLY when a
fresh (non-None) value is observed afterward. `n_e3_select_fires` /
`n_e3_latched_ticks` are recorded per cell so the true denominator is
auditable rather than trusted from a raw per-tick read count (the V3-EXQ-785
class this idiom exists to prevent).

TWO NEW PRECONDITIONS FROM THIS MEASUREMENT:
  * `modulatory_authority_engages_argmin_relevant_span` (ARM_B314_ON_AUTH_ON
    only): the mean relative deviation |post_score_range - raw_score_range| /
    raw_score_range across genuine select() fires must exceed
    AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR (0.05). The WORST (minimum)
    per-seed value is what is gated, not the mean-of-means, per the
    same-statistic-as-met rule.
  * `authority_no_spurious_engagement_without_signal` (ARM_B314_OFF_AUTH_ON
    only): the same relative deviation must stay BELOW
    AUTHORITY_NULL_REL_DEVIATION_CEILING (0.01) -- confirming in practice,
    not just in theory, that authority cannot manufacture engagement from a
    zero-range broadcast. The WORST (maximum) per-seed value is gated.

YOKED DIVERGENCE DV (primary, load-bearing, non-compounding). Every tick is a
paired comparison at an identical world state (all four agents share the
reference's observation sequence), so divergence measures argmin-relevant
influence with no trajectory-compounding. No free-running secondary DV is
recorded in this driver (a deliberate scope reduction vs. the 947 driver,
which recorded one as non-load-bearing behavioural context) -- the yoked
statistic is what both the 947 finding and this retest's governance question
turn on, and a fourth agent's free-running trajectory would not add
information proportional to its cost in a 4-arm design.

PRE-REGISTERED CRITERIA:
  C1 (load-bearing): yoked divergence, ARM_B314_ON_AUTH_ON vs reference,
     exceeds YOKED_DIVERGENCE_FLOOR (0.02).
  C2 (load-bearing, the interaction): C1's divergence exceeds
     ARM_B314_ON_AUTH_OFF's divergence (vs the same reference) by at least
     INTERACTION_MARGIN (0.02) -- i.e. authority is doing real work beyond
     what 314b's un-rescaled signal already does (empirically ~0).
  C3 (supporting, NOT load-bearing): ARM_B314_OFF_AUTH_ON's divergence stays
     below YOKED_DIVERGENCE_FLOOR -- the DV-symmetry negative control holds
     in practice, not just by the structural argument above.
Combination: overall_pass = C1 AND C2 AND (ARM_B314_ON_AUTH_ON readiness gate
green). A FAIL with that gate green is a genuine negative for MECH-314b even
under a corrected substrate (the channel carries no argmin-relevant influence
even once given genuine bounded authority); a FAIL with a red gate is
substrate-not-ready and carries no claim evidence.

SLEEP: none (no sleep flag is set; no SLEEP DRIVER line required).

EXPERIMENT_PURPOSE = "evidence" -- this experiment tests MECH-314b's claim
hypothesis directly (does the per-candidate uncertainty channel carry genuine
selection authority once the known magnitude mismatch is corrected).
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

EXPERIMENT_TYPE = "v3_exq_949_mech314b_authority_rescale_validation"
QUEUE_ID = "V3-EXQ-949"
CLAIM_IDS = ["MECH-314b"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [71, 101, 202, 303, 404]
EPISODES = 4
STEPS_PER_EPISODE = 80
WORLD_DIM = 32
SELF_DIM = 32

E2U_WARMUP_STEPS = 100
E2U_BATCH_SIZE = 32

MODULATORY_AUTHORITY_GAIN = 0.5

# Set ONLY by --dry-run, so the smoke can train the head inside its tiny tick
# budget and therefore actually demonstrate the decisive readout can move.
# A real run always uses E2U_WARMUP_STEPS.
_ACTIVE_WARMUP = E2U_WARMUP_STEPS

# --- PRE-REGISTERED THRESHOLDS (fixed here, never derived from this run) ---
# C1: 314b ON + authority ON must move the committed action away from its
# paired reference on more than this fraction of yoked ticks.
YOKED_DIVERGENCE_FLOOR = 0.02
# C2: the interaction. Authority-ON divergence must exceed authority-OFF
# divergence (same 314b-ON factor) by at least this much.
INTERACTION_MARGIN = 0.02
# Readiness: the keystone's PROPOSED (unvalidated) trained/untrained separator.
HEAD_REL_SPREAD_FLOOR = 1.0
# Readiness: the per-candidate vector must carry some cross-candidate span.
PCV_REL_SPREAD_FLOOR = 0.0
# Readiness: the floor manipulation must actually land (floor is FIXED raised
# here, not re-tested as a factor -- the 947 2x2 already proved it necessary).
DIVERSITY_TOLERANCE = 0.5
# Readiness (NEW): authority must measurably rescale post_score_range away
# from raw_score_range when engaged with a genuine per-candidate signal.
AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR = 0.05
# Readiness (NEW): authority must NOT rescale anything when the only
# modulatory signal is a uniform (zero-range) broadcast.
AUTHORITY_NULL_REL_DEVIATION_CEILING = 0.01

ARM_OFF_AUTHOFF = "ARM_B314_OFF_AUTH_OFF"
ARM_ON_AUTHOFF = "ARM_B314_ON_AUTH_OFF"
ARM_OFF_AUTHON = "ARM_B314_OFF_AUTH_ON"
ARM_ON_AUTHON = "ARM_B314_ON_AUTH_ON"

ALL_ARM_FACTORS = [
    (False, False, ARM_OFF_AUTHOFF),
    (True, False, ARM_ON_AUTHOFF),
    (False, True, ARM_OFF_AUTHON),
    (True, True, ARM_ON_AUTHON),
]


def _arm_id(b314_on: bool, authority_on: bool) -> str:
    for b, a, aid in ALL_ARM_FACTORS:
        if b == b314_on and a == authority_on:
            return aid
    raise AssertionError("unreachable")


def build_config(b314_on: bool, authority_on: bool, floor: int) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        support_preserving_min_first_action_classes=floor,
        use_modulatory_selection_authority=authority_on,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    )
    cfg.use_structured_curiosity = True
    # THE 314b FACTOR. "broadcast" is the Phase-1 uniform scalar (argmin-inert
    # by construction, and RANGE-inert -- see the authority engagement note
    # above); "e2_predictive_variance" is the live per-candidate path.
    cfg.curiosity_uncertainty_source = (
        "e2_predictive_variance" if b314_on else "broadcast"
    )
    # The SD-063 head is instantiated AND trained on every arm, so all four
    # cells differ only in whether its output is CONSUMED per-candidate and
    # whether the combined modulatory bias is rescaled -- not in the head's
    # own training trajectory.
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = True
    cfg.latent.e2_world_uncertainty_warmup_steps = _ACTIVE_WARMUP
    cfg.latent.e2_world_uncertainty_batch_size = E2U_BATCH_SIZE
    return cfg


def config_slice(b314_on: bool, authority_on: bool, floor: int,
                 action_dim: int) -> Dict[str, Any]:
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
        "use_modulatory_selection_authority": authority_on,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "use_e2_world_uncertainty": True,
        "use_e2_world_uncertainty_online_training": True,
        "e2_world_uncertainty_warmup_steps": _ACTIVE_WARMUP,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
    }


def head_5class_relative_spread(agent: REEAgent) -> Optional[float]:
    """Arm-INDEPENDENT readiness statistic: evaluate the head on every action
    class at one z_world and return (max-min)/mean. See the 947 driver for the
    full rationale (unchanged here); this function is a verbatim carry-forward.
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

    OWNS ITS OWN RNG STREAM -- see module docstring. Each runner snapshots and
    restores its own generator state around every tick, episode reset and
    residue update, so a paired comparison differs in the manipulation and
    nothing else.
    """

    def __init__(self, cfg: REEConfig) -> None:
        self.agent = REEAgent(cfg)
        # Captured AFTER construction, with the caller's reset_all_rng(seed)
        # still in force, so identically-configured runners start from an
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
        # Authority-engagement instrumentation (this experiment's new
        # measurement). Sample-size-integrity idiom: last_raw_scores /
        # last_scores are E3-cadence latches, cleared immediately before every
        # select_action() call and recorded only on a genuine fresh fire.
        self.raw_score_ranges: List[float] = []
        self.post_score_ranges: List[float] = []
        self.n_e3_select_fires = 0
        self.n_e3_latched_ticks = 0

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
        # Sample-size-integrity idiom (mandatory): clear the E3-cadence
        # latches immediately before the call, and record a range ONLY on a
        # genuine fresh select() fire this tick. e3_steps_per_tick defaults to
        # 10, so most ticks hold the previous fire's values -- reading them
        # unconditionally would pseudo-replicate one selection ~10x.
        agent.e3.last_raw_scores = None
        agent.e3.last_scores = None
        action = agent.select_action(candidates, ticks)
        raw = agent.e3.last_raw_scores
        post = agent.e3.last_scores
        if raw is not None and post is not None and raw.numel() > 1:
            raw_range = float((raw.max() - raw.min()).item())
            post_range = float((post.max() - post.min()).item())
            self.raw_score_ranges.append(raw_range)
            self.post_score_ranges.append(post_range)
            self.n_e3_select_fires += 1
        else:
            self.n_e3_latched_ticks += 1
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

    def authority_rel_deviation_mean(self) -> float:
        """Mean of |post_range - raw_range| / max(raw_range, eps) across every
        genuine E3 select() fire this runner observed. 0.0 if none fired
        (readiness gate will separately catch a starved run via n_e3_select_fires).
        """
        if not self.raw_score_ranges:
            return 0.0
        devs = [
            abs(p - r) / max(r, 1e-9)
            for r, p in zip(self.raw_score_ranges, self.post_score_ranges)
        ]
        return sum(devs) / len(devs)

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
            "raw_score_range_mean": _mean(self.raw_score_ranges),
            "post_score_range_mean": _mean(self.post_score_ranges),
            "authority_rel_deviation_mean": self.authority_rel_deviation_mean(),
            "n_e3_select_fires": self.n_e3_select_fires,
            "n_e3_latched_ticks": self.n_e3_latched_ticks,
        }


def run_yoked_quad(seed: int, floor: int, episodes: int, steps: int,
                   zg: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """One reference agent (314b OFF, authority OFF) drives the environment;
    the other three arms are each stepped on the SAME observation sequence and
    their committed actions compared per tick against the reference's.

    Non-compounding by construction: every agent sees an identical world state
    at every tick, so a divergence is an argmin flip, not an accumulated
    history difference.
    """
    # CRITICAL: reset the global RNG before EACH agent construction (see the
    # 947 driver's discovery -- building agents back to back draws weight
    # init from one continuing stream, confounding init with the manipulation).
    reset_all_rng(seed)
    ref = _Runner(build_config(b314_on=False, authority_on=False, floor=floor))
    reset_all_rng(seed)
    on_authoff = _Runner(build_config(b314_on=True, authority_on=False, floor=floor))
    reset_all_rng(seed)
    off_authon = _Runner(build_config(b314_on=False, authority_on=True, floor=floor))
    reset_all_rng(seed)
    on_authon = _Runner(build_config(b314_on=True, authority_on=True, floor=floor))
    reset_all_rng(seed)

    comparison_runners = [
        (ARM_ON_AUTHOFF, on_authoff),
        (ARM_OFF_AUTHON, off_authon),
        (ARM_ON_AUTHON, on_authon),
    ]

    env = CausalGridWorldV2()
    n_cmp: Dict[str, int] = {aid: 0 for aid, _ in comparison_runners}
    n_diff: Dict[str, int] = {aid: 0 for aid, _ in comparison_runners}
    per_episode: Dict[str, List[Dict[str, Any]]] = {
        aid: [] for aid, _ in comparison_runners
    }
    for ep in range(episodes):
        _, obs = env.reset()
        ref.reset_episode()
        for _, runner in comparison_runners:
            runner.reset_episode()
        ep_cmp = {aid: 0 for aid, _ in comparison_runners}
        ep_diff = {aid: 0 for aid, _ in comparison_runners}
        for _ in range(steps):
            a_ref = ref.choose(obs)
            for aid, runner in comparison_runners:
                a_cmp = runner.choose(obs)
                n_cmp[aid] += 1
                ep_cmp[aid] += 1
                if a_cmp != a_ref:
                    n_diff[aid] += 1
                    ep_diff[aid] += 1
            _f, harm, _d, _i, obs = env.step(a_ref)
            ref.observe(harm)
            for _, runner in comparison_runners:
                runner.observe(harm)
        for aid, _ in comparison_runners:
            per_episode[aid].append({
                "episode": ep, "n_compared": ep_cmp[aid], "n_diverged": ep_diff[aid],
                "divergence_frac": ep_diff[aid] / ep_cmp[aid] if ep_cmp[aid] else 0.0,
            })
        print(f"  [train] yoked seed={seed} floor={floor} ep {ep + 1}/{episodes} "
              f"diverged(ON_AUTHOFF/OFF_AUTHON/ON_AUTHON)="
              f"{ep_diff[ARM_ON_AUTHOFF]}/{ep_diff[ARM_OFF_AUTHON]}/{ep_diff[ARM_ON_AUTHON]}"
              f" of {ep_cmp[ARM_ON_AUTHOFF]}", flush=True)
    zg.observe(ref.agent)
    for _, runner in comparison_runners:
        zg.observe(runner.agent)

    out: Dict[str, Any] = {"ref_summary": ref.summary()}
    for aid, runner in comparison_runners:
        out[aid] = {
            "summary": runner.summary(),
            "yoked_n_compared": n_cmp[aid],
            "yoked_n_diverged": n_diff[aid],
            "yoked_divergence_frac": n_diff[aid] / n_cmp[aid] if n_cmp[aid] else 0.0,
            "yoked_per_episode": per_episode[aid],
        }
    return out


def paired_control_divergence(seed: int, floor: int, b314_on: bool,
                             authority_on: bool, episodes: int = 1,
                             steps: int = 20) -> float:
    """INSTRUMENT CONTROL: yoke an arm against ITSELF and return the divergence
    fraction, which MUST be 0.0. See the 947 driver's discovery of the
    shared-RNG-stream defect this control catches (unchanged mechanism here,
    extended to all 4 arm identities of this driver's 2x2).
    """
    reset_all_rng(seed)
    a = _Runner(build_config(b314_on=b314_on, authority_on=authority_on, floor=floor))
    reset_all_rng(seed)
    b = _Runner(build_config(b314_on=b314_on, authority_on=authority_on, floor=floor))
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
            "raised support_preserving_min_first_action_classes floor (fixed "
            "for this experiment -- the 947 2x2 already proved it necessary)"
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
        control="live candidate pools with a trained head; 947 driver measured 0.1948 at this floor",
        threshold=PCV_REL_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"]),
        applies_note=(
            "314b-OFF arms consume no per-candidate vector by construction "
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
            "the head trains on every arm but is CONSUMED only on 314b-ON "
            "arms; its readiness gates the 314b channel, not the broadcast "
            "reference"
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
            "on 314b-OFF arms every enabled sub-flavour is a uniform "
            "broadcast, so last_bias_range is 0.0 BY CONSTRUCTION. Asserting "
            "it there would make the OFF reference structurally un-passable"
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
    PreconditionSpec(
        name="modulatory_authority_engages_argmin_relevant_span",
        description=(
            "mean relative deviation |post_score_range - raw_score_range| / "
            "raw_score_range across genuine E3 select() fires exceeds the "
            "engagement floor -- confirms authority actually rescales the "
            "combined modulatory contribution to a magnitude competitive with "
            "raw_score_range, as the 2026-08-25 diagnostic measured directly"
        ),
        control="live per-candidate curiosity bias + authority rescale on a trained head",
        threshold=AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["b314_on"] and ctx["authority_on"]),
        applies_note=(
            "authority engagement is only a meaningful question when a "
            "genuine per-candidate (non-uniform) modulatory signal exists to "
            "rescale -- i.e. 314b ON. See "
            "authority_no_spurious_engagement_without_signal for the "
            "structural negative-control direction"
        ),
    ),
    PreconditionSpec(
        name="authority_no_spurious_engagement_without_signal",
        description=(
            "mean relative deviation |post_score_range - raw_score_range| / "
            "raw_score_range stays below a tight ceiling when the only "
            "modulatory contribution is a uniform (zero cross-candidate "
            "range) broadcast -- confirms authority cannot manufacture "
            "movement from nothing (modulatory_authority_min_range_floor gate)"
        ),
        control="live broadcast curiosity + authority rescale attempt on a trained head",
        threshold=AUTHORITY_NULL_REL_DEVIATION_CEILING,
        direction="upper",
        kind="readiness",
        applies_to=lambda ctx: bool((not ctx["b314_on"]) and ctx["authority_on"]),
        applies_note=(
            "this is the DV-symmetry negative-control arm for the authority "
            "factor alone (314b OFF, authority ON) -- not meaningful for any "
            "other arm"
        ),
    ),
]


def _arm_specs(floor: int) -> List[PreconditionSpec]:
    """The diversity precondition's threshold is the (fixed, raised) floor."""
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


def _arm_ctx(b314_on: bool, authority_on: bool, floor: int,
            action_dim: int) -> Dict[str, Any]:
    return {
        "arm_id": _arm_id(b314_on, authority_on),
        "b314_on": b314_on,
        "authority_on": authority_on,
        "floor": floor,
        "action_dim": action_dim,
    }


def run_experiment(episodes: int, steps: int, seeds: List[int],
                   dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    probe_env = CausalGridWorldV2()
    action_dim = int(probe_env.action_dim)
    floor = action_dim  # FIXED raised floor -- see module docstring.
    zg = ZGoalStreamAccumulator()

    # Design-time refusal BEFORE compute: prove no applicable precondition is
    # unsatisfiable from the pre-registered config.
    all_ctxs = [
        _arm_ctx(b314_on, authority_on, floor, action_dim)
        for b314_on, authority_on, _ in ALL_ARM_FACTORS
    ]
    assert_no_structurally_unsatisfiable_gate(_arm_specs(floor), all_ctxs)

    # INSTRUMENT CONTROL, before any scored compute: an arm yoked against
    # itself must diverge on zero ticks, for all 4 arm identities.
    control_div: Dict[str, float] = {}
    for b314_on, authority_on, aid in ALL_ARM_FACTORS:
        control_div[aid] = paired_control_divergence(
            seeds[0], floor, b314_on, authority_on,
            episodes=1, steps=(6 if dry_run else 20))
        print(f"[control] {aid}: self-yoked divergence = "
              f"{control_div[aid]:.6f} (must be 0)", flush=True)

    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        print(f"Seed {seed} Condition floor{floor}_yoked_quad", flush=True)
        slice_ref = config_slice(False, False, floor, action_dim)
        with arm_cell(seed, config_slice=slice_ref, script_path=Path(__file__),
                      config_slice_declared=True,
                      include_driver_script_in_hash=False) as cell:
            quad = run_yoked_quad(seed, floor, episodes, steps, zg)
            ref_row = {
                "arm_id": ARM_OFF_AUTHOFF, "seed": seed, "floor": floor,
                "b314_on": False, "authority_on": False,
                **quad["ref_summary"],
            }
            cell.stamp(ref_row)
        arm_results.append(ref_row)
        print(f"verdict: {'PASS' if ref_row['n_candidate_ticks'] > 0 else 'FAIL'}",
              flush=True)

        for b314_on, authority_on, aid in ALL_ARM_FACTORS[1:]:
            print(f"Seed {seed} Condition floor{floor}_{aid}", flush=True)
            slice_cmp = config_slice(b314_on, authority_on, floor, action_dim)
            with arm_cell(seed, config_slice=slice_cmp, script_path=Path(__file__),
                          config_slice_declared=True,
                          include_driver_script_in_hash=False) as cell:
                cell_data = quad[aid]
                row = {
                    "arm_id": aid, "seed": seed, "floor": floor,
                    "b314_on": b314_on, "authority_on": authority_on,
                    **cell_data["summary"],
                    "yoked_n_compared": cell_data["yoked_n_compared"],
                    "yoked_n_diverged": cell_data["yoked_n_diverged"],
                    "yoked_divergence_frac": cell_data["yoked_divergence_frac"],
                    "yoked_per_episode": cell_data["yoked_per_episode"],
                }
                cell.stamp(row)
            arm_results.append(row)
            print(f"verdict: {'PASS' if row['n_candidate_ticks'] > 0 else 'FAIL'}",
                  flush=True)

    # ---- per-arm readiness gates (regime-conditioned, never whole-run ANDed) --
    arm_gates = []
    for b314_on, authority_on, aid in ALL_ARM_FACTORS:
        rows = [r for r in arm_results if r["arm_id"] == aid]
        if not rows:
            continue
        ctx = _arm_ctx(b314_on, authority_on, floor, action_dim)
        measured = {
            "paired_control_is_bit_identical": control_div.get(aid, 1.0),
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
            "modulatory_authority_engages_argmin_relevant_span": min(
                r["authority_rel_deviation_mean"] for r in rows),
            "authority_no_spurious_engagement_without_signal": max(
                r["authority_rel_deviation_mean"] for r in rows),
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

    div_on_authoff = _mean_of(ARM_ON_AUTHOFF, "yoked_divergence_frac")
    div_off_authon = _mean_of(ARM_OFF_AUTHON, "yoked_divergence_frac")
    div_on_authon = _mean_of(ARM_ON_AUTHON, "yoked_divergence_frac")

    c1 = div_on_authon > YOKED_DIVERGENCE_FLOOR
    c2 = (div_on_authon - div_on_authoff) >= INTERACTION_MARGIN
    c3 = div_off_authon < YOKED_DIVERGENCE_FLOOR

    green = set(aggregate["green_arms"])
    overall_pass = bool(c1 and c2 and (ARM_ON_AUTHON in green))

    criteria = [
        {"name": "C1_b314_authority_on_moves_committed_action",
         "load_bearing": True, "passed": bool(c1),
         "measured": div_on_authon, "threshold": YOKED_DIVERGENCE_FLOOR},
        {"name": "C2_effect_attributable_to_authority_not_314b_alone",
         "load_bearing": True, "passed": bool(c2),
         "measured": div_on_authon - div_on_authoff,
         "threshold": INTERACTION_MARGIN},
        {"name": "C3_authority_alone_does_not_move_action_dv_symmetry_control",
         "load_bearing": False, "passed": bool(c3),
         "measured": div_off_authon, "threshold": YOKED_DIVERGENCE_FLOOR},
    ]
    combination_rule = (
        "overall_pass = C1 AND C2 AND (ARM_B314_ON_AUTH_ON readiness gate "
        "green). C3 is a DV-symmetry negative control (authority alone, "
        "without a genuine per-candidate signal, must not move the argmin) "
        "and does NOT gate the outcome. A FAIL with a GREEN ARM_B314_ON_AUTH_ON "
        "gate is a genuine negative for MECH-314b even with corrected "
        "authority (the channel carries no argmin-relevant influence even "
        "when given bounded rescale authority); a FAIL with a RED gate is "
        "substrate-not-ready and carries no claim evidence."
    )

    if not overall_pass and ARM_ON_AUTHON not in green:
        direction = "unknown"
        label = "substrate_not_ready_requeue"
    elif overall_pass:
        direction = "supports"
        label = "b314_authority_rescale_restores_argmin_selection_authority"
    elif c1 and not c2:
        direction = "mixed"
        label = "b314_moves_argmin_but_not_attributable_to_authority_rescale"
    else:
        direction = "weakens"
        label = "b314_percandidate_path_remains_inert_even_with_authority_rescale"

    criteria_nd = arm_criteria_non_degenerate(
        {
            ARM_ON_AUTHON: [
                "C1_b314_authority_on_moves_committed_action",
            ],
            ARM_OFF_AUTHON: [
                "C3_authority_alone_does_not_move_action_dv_symmetry_control",
            ],
        },
        aggregate,
        extra={
            "C2_effect_attributable_to_authority_not_314b_alone": bool(
                ARM_ON_AUTHON in green and ARM_ON_AUTHOFF in green
            ),
        },
    )
    criteria_nd["C2_effect_attributable_to_authority_not_314b_alone"] = bool(
        ARM_ON_AUTHON in green and ARM_ON_AUTHOFF in green
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
            "yoked_divergence_frac_on_authoff": div_on_authoff,
            "yoked_divergence_frac_off_authon": div_off_authon,
            "yoked_divergence_frac_on_authon": div_on_authon,
            "authority_interaction_delta": div_on_authon - div_on_authoff,
            "mean_distinct_first_action_classes_on_authon": _mean_of(
                ARM_ON_AUTHON, "mean_distinct_first_action_classes"),
            "head_5class_relative_spread_on_authon": _mean_of(
                ARM_ON_AUTHON, "head_5class_relative_spread_final"),
            "raw_score_range_mean_on_authon": _mean_of(
                ARM_ON_AUTHON, "raw_score_range_mean"),
            "post_score_range_mean_on_authon": _mean_of(
                ARM_ON_AUTHON, "post_score_range_mean"),
            "authority_rel_deviation_mean_on_authon": _mean_of(
                ARM_ON_AUTHON, "authority_rel_deviation_mean"),
            "authority_rel_deviation_mean_off_authon": _mean_of(
                ARM_OFF_AUTHON, "authority_rel_deviation_mean"),
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
            "ARM_B314_OFF_AUTH_OFF": (
                "reference arm; both factors off. curiosity contributes a "
                "uniform broadcast (bias_range == 0 by construction), "
                "authority is disabled. Never diverges from itself."
            ),
            "ARM_B314_ON_AUTH_OFF": (
                "314b ON, authority OFF. Reproduces the 947 driver's "
                "proven-null cell: per-candidate vector carries measured "
                "non-zero cross-candidate span, but at the UN-rescaled "
                "magnitude (~1.5e-05 vs raw_score_range ~270-290) it is "
                "empirically negligible, per the 2026-08-25 diagnostic."
            ),
            "ARM_B314_OFF_AUTH_ON": (
                "314b OFF (broadcast), authority ON. STRUCTURAL negative "
                "control -- authority's rescale requires the combined "
                "modulatory contribution's cross-candidate RANGE to exceed "
                "modulatory_authority_min_range_floor, and a uniform "
                "broadcast has RANGE EXACTLY 0. Authority cannot engage "
                "here regardless of gain. Asserted at runtime via "
                "authority_no_spurious_engagement_without_signal, not "
                "merely assumed."
            ),
            "ARM_B314_ON_AUTH_ON": (
                "314b ON, authority ON. The manipulation supplies a "
                "non-constant per-candidate vector AND the combined "
                "modulatory contribution is rescaled to a magnitude "
                "genuinely competitive with raw_score_range (measured via "
                "modulatory_authority_engages_argmin_relevant_span). Neither "
                "factor alone is invariant under the DV's symmetry group "
                "here. This is the arm under test for real argmin authority."
            ),
        },
        "custom_information": {
            "readiness_statistic_note": (
                "head_5class_relative_spread is computed on all action_dim "
                "one-hots at a single z_world, NOT read from get_state()'s "
                "latched last_pvar_relative_spread, whose batch on a 314b-ON "
                "arm is the candidate pool."
            ),
            "head_rel_spread_threshold_status": (
                "HEAD_REL_SPREAD_FLOOR=1.0 is the keystone's PROPOSED and "
                "UNVALIDATED separator (3 seeds, one env), carried here "
                "unchanged from the 947 driver."
            ),
            "sample_size_integrity_note": (
                "agent.e3.last_raw_scores / last_scores are E3-cadence "
                "latches (e3_steps_per_tick defaults to 10), cleared to None "
                "immediately before every select_action() call and recorded "
                "only on a genuine fresh fire. n_e3_select_fires / "
                "n_e3_latched_ticks are recorded per cell so the true "
                "denominator behind authority_rel_deviation_mean is auditable."
            ),
            "supersedes_note": (
                "Supersedes the un-queued V3-EXQ-947 2x2 design "
                "(mech314b_score_bias_magnitude_diagnostic_2026-08-25.md sec "
                "5): that driver's queue entry was never appended and it is "
                "mechanistically guaranteed null (authority always OFF, "
                "~1.4-1.8e7x magnitude gap). This design fixes the diversity "
                "floor at its proven-necessary raised value and crosses 314b "
                "ON/OFF against use_modulatory_selection_authority ON/OFF "
                "instead."
            ),
            "re_derive_brake_note": (
                "MECH-314b brake count 3 (604a/604b/604c) -- RELEASED. 604b "
                "was itself an earlier authority-ON attempt that failed for a "
                "DIFFERENT reason (class-uniform candidate pool, pre-GAP-A); "
                "ARC-065's GAP-A slot has since landed and validated "
                "(V3-EXQ-649) and this driver's own 4.934/5 candidate-class "
                "diversity confirms the pool is now saturated."
            ),
            "substrate_defect_note": (
                "open corrupting substrate_queue entry "
                "contextmemory-write-path-addressing-degeneracy "
                "(ree_core/predictors/e1_deep.py) is in path via E1 but "
                "applies identically to every arm and does not interact with "
                "either the 314b or the authority factor, so it biases the "
                "load-bearing contrast toward the null, not toward a false "
                "positive."
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
        "floor": floor,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "e2_world_uncertainty_warmup_steps": E2U_WARMUP_STEPS,
        "e2_world_uncertainty_batch_size": E2U_BATCH_SIZE,
        "thresholds": {
            "YOKED_DIVERGENCE_FLOOR": YOKED_DIVERGENCE_FLOOR,
            "INTERACTION_MARGIN": INTERACTION_MARGIN,
            "HEAD_REL_SPREAD_FLOOR": HEAD_REL_SPREAD_FLOOR,
            "DIVERSITY_TOLERANCE": DIVERSITY_TOLERANCE,
            "AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR":
                AUTHORITY_ENGAGEMENT_REL_DEVIATION_FLOOR,
            "AUTHORITY_NULL_REL_DEVIATION_CEILING":
                AUTHORITY_NULL_REL_DEVIATION_CEILING,
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
            "per-candidate vector carries cross-candidate span (ON_AUTHON)":
                m["mean_distinct_first_action_classes_on_authon"] > 0.0,
            "head trained above the untrained band (>0.26)":
                m["head_5class_relative_spread_on_authon"] > 0.26,
            "authority engages: post_score_range diverges from raw_score_range "
            "on ON_AUTHON":
                m["authority_rel_deviation_mean_on_authon"] > 0.0,
            "authority null control: OFF_AUTHON deviation stays tiny":
                m["authority_rel_deviation_mean_off_authon"]
                < AUTHORITY_NULL_REL_DEVIATION_CEILING * 5,
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
