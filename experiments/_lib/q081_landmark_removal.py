"""
q081_landmark_removal -- the STRUCTURE-DESTROYING arm required by Q-081's
non-degeneracy guard, alongside (never instead of) the constrained-realisation
surrogate in `q081_surrogate.py`.

WHAT THIS IS FOR
----------------
Q-081 asks whether REE's configured multi-rate execution (SD-006: E1/E2/E3 at
1/3/10) produces SHARED cross-stream organisation (Outcome A) or only the rate
separation and wired gates it was configured with (Outcome B). Two controls are
required and they are NOT substitutes for one another:

  * `q081_surrogate.py` destroys cross-stream alignment in the ANALYSIS -- a
    block permutation within each stream's own tick grid, preserving tick times,
    marginals and within-stream autocorrelation.
  * THIS MODULE destroys the event / commitment LANDMARK structure in the
    SYSTEM, while leaving intact (a) the streams, (b) the configured update
    rates, and (c) the environmental input statistics.

A cross-stream statistic that survives THIS arm was measuring the clock. That is
the sharpest available A-vs-B discriminator, and it is why clearing the surrogate
null is necessary and nowhere near sufficient -- wired coordination is real
coordination and will correctly clear any surrogate test.

The worked analogue is Chang, Nastase & Hasson 2022 (PNAS): the scrambled-story
control preserved low-level acoustic and lexical input statistics, destroyed only
the nested event structure, and the cross-area lag gradient vanished -- which is
what excluded the intrinsic-rate explanation. See
REE_assembly/evidence/literature/targeted_review_mech_466/entries/
2026-07-22_mech_466_cross_timescale_lag_gradient_scrambled_control_chang2022/.

WHERE IT INTERVENES, AND WHY NO ree_core CHANGE IS NEEDED
---------------------------------------------------------
`REEAgent.sense()` calls `hippocampal.event_segmenter.step(...)` once per waking
tick and binds the result to a single local `events` list, which is then the sole
input to ALL THREE live consumer paths (agent.py ~4249-4367):

    events = self.hippocampal.event_segmenter.step(...)
      -> self.hippocampal._boundary_event_queue.extend(events)          # MECH-288
      -> self.hippocampal.invalidation_trigger.step(boundary_events=events)
                                                                        # MECH-287
         -> broadcasts -> apply_invalidation_broadcasts_to_regions()
            (drops per-region V_s entries, marks anchors inactive)
      -> self.hippocampal.tick_anchor_set(events=anchor_events)         # MECH-269
         -> anchor_set.consume_boundary_events() -> write_anchor(
                scale, segment_id_new, stream_mixture, z_world)

That local list is a single choke point, so wrapping `event_segmenter.step` is
sufficient and complete. This follows the established experiment-layer precedent
(`consolidation_lesion_harness.py`, V3-EXQ-702 injected content): no ree_core
change, no new REEConfig knob, no backward-compatibility surface, and the lever
is removed entirely by `detach()`.

Note the third consumer is why the arm has behavioural reach at all:
`write_anchor` binds `segment_id_new` to the z_world present AT EMISSION TIME. Re-
emitting a landmark at a decorrelated tick therefore binds the anchor to a
different world state -- the alignment is destroyed in the system, not merely
relabelled. Anchors feed V_s, V_s feeds `vs_rollout_gate`, and the gate feeds E3
selection.

BEHAVIOURAL REACH IS FLAG-CONDITIONAL -- ASSERT IT, DO NOT ASSUME IT
--------------------------------------------------------------------
If `use_invalidation_trigger`, `use_anchor_sets` and `use_per_region_vs` are all
False, boundary events are queued and drained with no consumer, so the arm is
behaviourally INERT: it then trivially preserves input statistics (there is no
path by which it could change them) but tests only statistics computed on the
boundary stream itself, which is not what Q-081 asks. `assert_behavioural_reach()`
makes this a precondition rather than a silent vacuity, in the spirit of the
MECH-466 non-degeneracy guard. An arm that cannot act is not a control.

THE HARD CONSTRAINT: PRESERVING INPUT STATISTICS
------------------------------------------------
This is the part that was hard for Chang and is harder here, and it is the reason
this module reports rather than merely intervenes.

Chang had it easy in one respect: the stimulus was EXOGENOUS, so scrambling could
not change what the subject received. REE is a CLOSED LOOP. Any intervention with
behavioural reach -- and per the section above, an intervention without
behavioural reach is vacuous -- propagates to action, and therefore to what the
agent encounters. Input statistics cannot be preserved by fiat in a closed loop.
Claiming otherwise would be the confound the arm exists to avoid.

So preservation is established at two levels, and the second one is a measurement,
not a guarantee:

  1. BY CONSTRUCTION AT THE INTERVENTION SITE. The emitted boundary train is a
     permutation of a donor train, so boundary COUNT, the inter-event-interval
     MULTISET, the POSTERIOR multiset (which sets broadcast_strength =
     posterior * gain) and the SCALE MIX are all preserved EXACTLY, not in
     distribution. `preservation_report()` audits each of these per episode and
     sets `preserved_by_construction=False` if any fails. Nothing downstream can
     tell the arm apart from the intact arm by the marginal statistics of the
     landmark signal itself -- only by its timing relative to system state.

  2. BY MEASUREMENT DOWNSTREAM. `input_statistics_divergence()` compares the
     scrambled arm against the intact arm on the quantities that define "what the
     agent encountered": per-channel observation marginals, state-visitation
     distribution, action distribution, harm/reward event rate, and episode
     length. The thresholds are PRE-REGISTERED and the verdict is
     `input_statistics_preserved` -- and when it is False the arm is CONFOUNDED
     and self-routes, exactly as a boundary-rate floor/ceiling pin self-routes
     substrate_not_ready under MECH-466. A confounded arm must not be reported as
     a null result: "the statistic vanished" and "the agent was somewhere else"
     are indistinguishable without this check.

  Level 2 can fail legitimately. That is a finding about coupling strength, not a
  harness bug, and it must be reported as such rather than tuned away.

THE LEVER: YOKED PERMUTATION OF THE DONOR TRAIN (primary)
----------------------------------------------------------
Three levers were considered. The choice is justified rather than assumed, per the
Q-081 guard's insistence that this surrogate be DESIGNED for REE.

  * REJECTED as primary -- SUPPRESS boundary emission (`mode="suppress"`). This
    removes landmark structure but also removes the drive: broadcast count goes to
    zero, so the arm confounds "landmarks misaligned" with "less invalidation
    drive overall". It is a LESION, not a scramble, and Chang's control was
    explicitly not a lesion. Retained as an out-of-family reference arm and
    labelled `is_lesion=True` so it can never be mistaken for the primary.

  * REJECTED as primary -- ONLINE RATE-MATCHED RESAMPLING. Emitting with
    probability p = running boundary rate preserves the mean rate but imposes
    geometric inter-event intervals. If the true boundary train is bursty (and
    hierarchical segmentation is bursty by construction -- a slow fire suppresses
    a fast fire on the same tick), the IEI distribution is NOT preserved, and a
    statistic could then die from the loss of burstiness rather than from the loss
    of alignment. That is the same class of error as the fresh-only shuffle the
    surrogate module supersedes: significance achieved by destroying something
    that was never under test.

  * ADOPTED -- YOKED PERMUTATION (`mode="iei_permute"`). The intact arm is run
    first and its boundary train recorded per (seed, episode). The scrambled arm
    replays a permutation of THAT EXACT train: the (interval, payload) pairs are
    permuted jointly as units, which is precisely the paragraph-scramble of the
    analogue -- same segments, same durations, same content, scrambled order and
    therefore scrambled alignment. Because the permuted intervals are a
    rearrangement of the same multiset, their cumulative sum ends at the donor's
    last boundary tick, so no event falls off the end of the episode and the count
    is preserved exactly rather than approximately. Matching is seed-for-seed and
    episode-for-episode, which makes the preservation guarantee auditable rather
    than statistical. This is a yoked-control design in the behavioural-
    neuroscience sense, and it inherits that design's known caveat: after the
    first divergence the donor train is no longer "what this agent would have
    produced". That is accepted deliberately -- an exactly-matched landmark signal
    is worth more here than a counterfactually-faithful one, because the whole
    question is whether alignment mattered.

  * ADOPTED as the conservative secondary -- `mode="circular_shift"`. A rigid
    circular shift of the donor train destroys landmark-to-stream alignment while
    preserving the landmark train's OWN internal structure (its autocorrelation,
    its burstiness, its interval SEQUENCE) up to a single wrap interval. Running
    it alongside `iei_permute` dissociates two things that the primary arm
    destroys together: alignment with the rest of the system, versus the internal
    organisation of the landmark train. A statistic that dies under `iei_permute`
    but survives `circular_shift` was reading landmark-train structure, not
    cross-stream alignment. That dissociation is worth the extra arm.

  * ADOPTED as the donor-free fallback -- `mode="jitter"`. Causal, needs no donor,
    preserves the count exactly (pending events are flushed at episode end), but
    smears the IEI distribution. Use it only when no donor is available; the
    preservation report will show `iei_multiset_match=False`, correctly.

WHAT THIS MODULE DOES NOT DELIVER
----------------------------------
It does not adjudicate Q-081. It is one of the two required controls, and even
both together are not the ablation series -- only that series separates Outcome A
from Outcome B. Do NOT report a surrogate-cleared statistic as evidence for
Outcome A on its own, and do not report a statistic that survives this arm as
evidence for anything except the clock.

It also does not scramble the SEGMENT-ID stream that other consumers read via
`event_segmenter.current_segment_id()`. The real segmenter keeps running and keeps
its own true counters; only the emitted EVENT stream is scrambled. With
`rewrite_segment_ids=True` (default) the emitted events carry a self-consistent
synthetic id sequence so consumers see a coherent segmentation that is merely
misaligned, rather than id references that jump.

Design record: REE_assembly/evidence/planning/q081_landmark_removal_arm_design.md
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LandmarkRemovalConfig",
    "RecordedBoundary",
    "BoundaryTrain",
    "LandmarkScrambler",
    "assert_behavioural_reach",
    "input_statistics_divergence",
    "MODES",
    "PRIMARY_MODE",
]

# Modes. "off" is the intact arm (pass-through recorder).
MODES = ("off", "iei_permute", "circular_shift", "jitter", "suppress")

# The pre-registered primary. Named so drivers and contracts agree without
# each restating the justification above.
PRIMARY_MODE = "iei_permute"

# Modes that remove the drive as well as the alignment. Never the primary.
_LESION_MODES = ("suppress",)

# Modes that replay a donor train and therefore preserve the count and the
# interval multiset exactly.
_DONOR_MODES = ("iei_permute", "circular_shift")


# --------------------------------------------------------------------------- #
# Pre-registered input-statistics thresholds                                   #
# --------------------------------------------------------------------------- #
# Fixed here, before any run, so the verdict cannot be tuned after seeing the
# result. These are deliberately loose: the arm is meant to catch a GROSS shift
# in what the agent encountered (it wandered somewhere else entirely), not to
# demand bit-equality, which a closed loop cannot deliver. A tight threshold
# would make every honest run look confounded.
DEFAULT_INPUT_STAT_THRESHOLDS: Dict[str, float] = {
    # Jensen-Shannon divergence, base 2, in [0, 1].
    "state_visitation_js": 0.10,
    "action_distribution_js": 0.10,
    # Absolute difference in per-episode event rate, events per step.
    "harm_rate_abs": 0.02,
    "reward_rate_abs": 0.02,
    # Relative difference in mean episode length.
    "episode_length_rel": 0.10,
    # Standardised difference in per-channel observation means, max over
    # channels, in units of the intact arm's pooled SD.
    "obs_channel_mean_std": 0.25,
}


# --------------------------------------------------------------------------- #
# Boundary train recording                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RecordedBoundary:
    """One boundary as recorded from the intact arm, payload preserved.

    `posterior` is carried verbatim because broadcast_strength = posterior * gain
    downstream: preserving the posterior multiset is what keeps the invalidation
    DRIVE matched between arms.
    """
    t: int
    scale: str
    posterior: float
    sources: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "t": int(self.t),
            "scale": str(self.scale),
            "posterior": float(self.posterior),
            "sources": list(self.sources),
        }


@dataclass
class BoundaryTrain:
    """The boundary train of one (seed, episode) of the intact arm.

    This is the donor object. It is JSON-round-trippable so an intact arm can be
    run, banked and consumed by a later scrambled arm without re-running it.
    """
    seed: int
    episode_index: int
    n_steps: int
    boundaries: List[RecordedBoundary] = field(default_factory=list)

    # -- construction ------------------------------------------------------- #

    def add(self, b: RecordedBoundary) -> None:
        self.boundaries.append(b)

    # -- derived quantities ------------------------------------------------- #

    @property
    def count(self) -> int:
        return len(self.boundaries)

    def times(self) -> List[int]:
        return [b.t for b in self.boundaries]

    def intervals(self) -> List[int]:
        """Inter-event intervals, with the first measured from t=0.

        Defining d_0 = t_0 makes the interval list a complete reparameterisation
        of the time list: cumsum(intervals) == times exactly. That is what lets a
        permutation of the intervals be guaranteed to land inside the episode.
        """
        out: List[int] = []
        prev = 0
        for b in self.boundaries:
            out.append(int(b.t) - prev)
            prev = int(b.t)
        return out

    def posteriors(self) -> List[float]:
        return [float(b.posterior) for b in self.boundaries]

    def scale_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for b in self.boundaries:
            out[b.scale] = out.get(b.scale, 0) + 1
        return out

    # -- serialisation ------------------------------------------------------ #

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "episode_index": int(self.episode_index),
            "n_steps": int(self.n_steps),
            "boundaries": [b.as_dict() for b in self.boundaries],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BoundaryTrain":
        return cls(
            seed=int(d["seed"]),
            episode_index=int(d["episode_index"]),
            n_steps=int(d["n_steps"]),
            boundaries=[
                RecordedBoundary(
                    t=int(b["t"]),
                    scale=str(b["scale"]),
                    posterior=float(b["posterior"]),
                    sources=tuple(b.get("sources", ())),
                )
                for b in d.get("boundaries", [])
            ],
        )


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class LandmarkRemovalConfig:
    """Arm configuration.

    mode:
      "off"            intact arm -- pass through, record the donor train.
      "iei_permute"    PRIMARY. Joint permutation of (interval, payload) units of
                       the donor train. Count / IEI multiset / posterior multiset
                       / scale mix preserved exactly.
      "circular_shift" Conservative secondary. Rigid circular shift of the donor
                       train; preserves the interval SEQUENCE up to one wrap.
      "jitter"         Donor-free fallback. Causal per-event delay; preserves the
                       count exactly, smears the IEI distribution.
      "suppress"       LESION reference, not the primary: removes the drive too.

    rewrite_segment_ids: emit a self-consistent synthetic segment-id sequence
      (mirroring the segmenter's own rule: a slow fire increments outer and
      resets inner; anything else increments inner) so consumers see a coherent
      but misaligned segmentation rather than id references that jump. The real
      segmenter's own counters are never touched.

    shift_steps: for "circular_shift". None -> drawn from the harness RNG in the
      middle half of the episode, which keeps the shift well away from both the
      no-op (0) and the near-no-op (n_steps).
    """
    mode: str = "off"
    seed: int = 0
    rewrite_segment_ids: bool = True
    # jitter
    jitter_min_steps: int = 1
    jitter_max_steps: int = 40
    # circular_shift
    shift_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(
                "LandmarkRemovalConfig.mode must be one of "
                + ", ".join(MODES)
                + "; got " + repr(self.mode)
            )
        if self.jitter_min_steps < 0:
            raise ValueError("jitter_min_steps must be >= 0")
        if self.jitter_max_steps < self.jitter_min_steps:
            raise ValueError("jitter_max_steps must be >= jitter_min_steps")

    @property
    def is_lesion(self) -> bool:
        """True for arms that remove the invalidation DRIVE, not only alignment.

        A lesion arm is a reference, never the primary discriminator: it
        confounds misaligned landmarks with fewer landmarks.
        """
        return self.mode in _LESION_MODES

    @property
    def requires_donor(self) -> bool:
        return self.mode in _DONOR_MODES

    @property
    def is_intact(self) -> bool:
        return self.mode == "off"

    def arm_id(self) -> str:
        return "q081_landmark_" + self.mode


# --------------------------------------------------------------------------- #
# The scrambler                                                                #
# --------------------------------------------------------------------------- #


class LandmarkScrambler:
    """Wraps `agent.hippocampal.event_segmenter.step` to scramble the emitted
    boundary train while leaving the segmenter's own detectors running.

    Lifecycle per episode:

        scr = LandmarkScrambler(cfg)
        scr.attach(agent)
        for ep in range(n_episodes):
            agent_reset_and_env_reset(...)
            scr.begin_episode(ep, n_steps=T, donor=donor_for(seed, ep))
            ... run the episode ...
            scr.end_episode()
        scr.detach()
        report = scr.preservation_report()

    `begin_episode` must be called AFTER the agent's per-episode reset, because
    that reset clears the segmenter (agent.py ~2914) and the harness mirrors the
    segmenter's per-episode clock.
    """

    def __init__(self, config: LandmarkRemovalConfig):
        self.config = config
        self._rng = random.Random(config.seed)
        self._agent: Optional[Any] = None
        self._segmenter: Optional[Any] = None
        self._orig_step: Optional[Any] = None
        self._step_was_instance_attr: bool = False

        # Per-episode state.
        self._episode_index: int = -1
        self._n_steps: int = 0
        self._donor: Optional[BoundaryTrain] = None
        self._true_train: Optional[BoundaryTrain] = None
        self._emitted_train: Optional[BoundaryTrain] = None
        # mode-specific schedules
        self._schedule: Dict[int, List[RecordedBoundary]] = {}
        self._pending: List[Tuple[int, Any]] = []  # (due_t, event) for jitter
        self._tick: int = 0
        # synthetic segment-id counters (mirror the segmenter's rule)
        self._outer: int = 0
        self._inner: int = 0
        self._slow_scale_name: str = "slow"
        self._id_format: str = "{outer}.{inner}"

        # Accumulated per-episode audits.
        self._episodes: List[Dict[str, Any]] = []
        self._donor_missing_episodes: List[int] = []
        # Donor trains banked by an intact ('off') run, keyed in order of
        # episode close. Consumed by a later scrambled arm via
        # recorded_trains() / donor_index().
        self._donor_bank: List[BoundaryTrain] = []

    # -- attach / detach ---------------------------------------------------- #

    def attach(self, agent: Any) -> None:
        """Install the wrapper. Idempotent-unsafe by design: double-attach raises
        rather than silently nesting two scramblers, which would be undetectable
        in the manifest.
        """
        if self._orig_step is not None:
            raise RuntimeError("LandmarkScrambler already attached; detach() first.")
        hippo = getattr(agent, "hippocampal", None)
        if hippo is None:
            raise RuntimeError(
                "LandmarkScrambler requires agent.hippocampal; got None. "
                "Enable the hippocampal module before attaching."
            )
        seg = getattr(hippo, "event_segmenter", None)
        if seg is None:
            raise RuntimeError(
                "LandmarkScrambler requires hippocampal.event_segmenter; got None. "
                "Set use_event_segmenter=True before attaching."
            )
        self._agent = agent
        self._segmenter = seg
        self._slow_scale_name = str(getattr(seg, "slow_scale_name", "slow"))
        self._id_format = str(getattr(seg, "scale_id_format", "{outer}.{inner}"))
        self._orig_step = seg.step
        # `step` is normally a class-level method, so assigning the wrapper
        # creates an INSTANCE attribute that did not exist before. detach() must
        # delete it rather than write the bound method back, or the segmenter is
        # left permanently shadowed by an instance attribute -- harmless to call
        # but a trap for any later patching, and a silent difference from a
        # never-attached segmenter.
        self._step_was_instance_attr = "step" in vars(seg)

        def _wrapped(latent_dict, pe_dict, t, _self=self, _orig=seg.step):
            true_events = _orig(latent_dict=latent_dict, pe_dict=pe_dict, t=t)
            return _self._on_step(int(t), list(true_events or []))

        seg.step = _wrapped

    def detach(self) -> None:
        """Remove the wrapper. Safe to call when not attached."""
        if self._segmenter is not None and self._orig_step is not None:
            if self._step_was_instance_attr:
                self._segmenter.step = self._orig_step
            else:
                # Restore the class method exactly, leaving no instance shadow.
                vars(self._segmenter).pop("step", None)
        self._segmenter = None
        self._orig_step = None
        self._agent = None

    @property
    def attached(self) -> bool:
        return self._orig_step is not None

    # -- episode lifecycle -------------------------------------------------- #

    def begin_episode(
        self,
        episode_index: int,
        n_steps: int,
        seed: int = 0,
        donor: Optional[BoundaryTrain] = None,
    ) -> None:
        if self._episode_index >= 0 and self._emitted_train is not None:
            # Defensive: a driver that forgot end_episode() would otherwise
            # silently lose an episode's audit.
            self.end_episode()
        self._episode_index = int(episode_index)
        self._n_steps = int(n_steps)
        self._donor = donor
        self._tick = 0
        self._pending = []
        self._schedule = {}
        self._outer = 0
        self._inner = 0
        self._true_train = BoundaryTrain(
            seed=int(seed), episode_index=int(episode_index), n_steps=int(n_steps)
        )
        self._emitted_train = BoundaryTrain(
            seed=int(seed), episode_index=int(episode_index), n_steps=int(n_steps)
        )

        if self.config.requires_donor:
            if donor is None:
                # Recorded, not raised: a missing donor in one episode should not
                # destroy an expensive run. It IS reported, and it forces
                # preserved_by_construction=False for the arm.
                self._donor_missing_episodes.append(int(episode_index))
            else:
                self._schedule = self._build_schedule(donor)

    def end_episode(self) -> Dict[str, Any]:
        """Close the episode, flush anything pending, and bank the audit."""
        audit = self._episode_audit()
        self._episodes.append(audit)
        if self.config.is_intact and self._true_train is not None:
            # Only the intact arm banks donors. See recorded_trains().
            self._donor_bank.append(self._true_train)
        self._episode_index = -1
        self._true_train = None
        self._emitted_train = None
        self._schedule = {}
        self._pending = []
        return audit

    # -- the intercept ------------------------------------------------------ #

    def _on_step(self, t: int, true_events: List[Any]) -> List[Any]:
        """Record the true fires, return the scrambled emission for this tick."""
        if self._true_train is None:
            # Not inside an episode (driver did not call begin_episode). Pass
            # through unchanged rather than silently scrambling an unaudited
            # stretch of ticks.
            return true_events

        for ev in true_events:
            self._true_train.add(
                RecordedBoundary(
                    t=int(t),
                    scale=str(getattr(ev, "scale", "")),
                    posterior=float(getattr(ev, "posterior", 0.0)),
                    sources=tuple(getattr(ev, "sources", ()) or ()),
                )
            )

        mode = self.config.mode
        if mode == "off":
            out = true_events
        elif mode == "suppress":
            out = []
        elif mode == "jitter":
            out = self._step_jitter(t, true_events)
        else:  # iei_permute | circular_shift -- schedule-driven
            out = self._step_scheduled(t)

        for ev in out:
            self._emitted_train.add(
                RecordedBoundary(
                    t=int(t),
                    scale=str(getattr(ev, "scale", "")),
                    posterior=float(getattr(ev, "posterior", 0.0)),
                    sources=tuple(getattr(ev, "sources", ()) or ()),
                )
            )
        self._tick = int(t)
        return out

    # -- mode implementations ----------------------------------------------- #

    def _build_schedule(self, donor: BoundaryTrain) -> Dict[int, List[RecordedBoundary]]:
        """Map emission tick -> boundaries to emit, from the donor train."""
        if donor.count == 0:
            return {}
        mode = self.config.mode
        if mode == "iei_permute":
            units = list(zip(donor.intervals(), donor.boundaries))
            self._rng.shuffle(units)
            sched: Dict[int, List[RecordedBoundary]] = {}
            cursor = 0
            for interval, b in units:
                cursor += int(interval)
                # cumsum of a permutation of the same multiset ends exactly at
                # the donor's last boundary tick, so no clipping is possible and
                # the count is preserved exactly. Clamp defensively anyway.
                tt = min(cursor, max(0, donor.n_steps - 1))
                sched.setdefault(tt, []).append(b)
            return sched
        if mode == "circular_shift":
            n = max(1, int(donor.n_steps))
            shift = self.config.shift_steps
            if shift is None:
                # Middle half: away from both the no-op and the near-no-op.
                lo, hi = n // 4, max(n // 4 + 1, (3 * n) // 4)
                shift = self._rng.randrange(lo, hi)
            sched = {}
            for b in donor.boundaries:
                tt = (int(b.t) + int(shift)) % n
                sched.setdefault(tt, []).append(b)
            return sched
        return {}

    def _step_scheduled(self, t: int) -> List[Any]:
        due = self._schedule.get(int(t), [])
        return [self._synthesise(b, t) for b in due]

    def _step_jitter(self, t: int, true_events: List[Any]) -> List[Any]:
        for ev in true_events:
            d = self._rng.randint(
                int(self.config.jitter_min_steps), int(self.config.jitter_max_steps)
            )
            self._pending.append((int(t) + int(d), ev))
        out: List[Any] = []
        still: List[Tuple[int, Any]] = []
        last_tick = max(0, int(self._n_steps) - 1)
        for due_t, ev in self._pending:
            if due_t <= t or (t >= last_tick):
                # Flush at (or past) the due tick; and flush everything on the
                # final tick so the emitted COUNT matches the true count exactly
                # rather than losing the tail. Tail-flushed events are counted in
                # `n_flushed_at_episode_end`.
                out.append(self._retime(ev, t))
            else:
                still.append((due_t, ev))
        self._pending = still
        return out

    # -- event construction -------------------------------------------------- #

    def _next_segment_ids(self, scale: str) -> Tuple[str, str]:
        old = self._id_format.format(outer=self._outer, inner=self._inner)
        if scale == self._slow_scale_name:
            self._outer += 1
            self._inner = 0
        else:
            self._inner += 1
        new = self._id_format.format(outer=self._outer, inner=self._inner)
        return old, new

    def _synthesise(self, b: RecordedBoundary, t: int) -> Any:
        """Build a BoundaryEvent carrying the donor payload at emission tick t."""
        from ree_core.hippocampal.event_segmenter import BoundaryEvent  # local
        if self.config.rewrite_segment_ids:
            old, new = self._next_segment_ids(b.scale)
        else:
            old = self._id_format.format(outer=self._outer, inner=self._inner)
            new = old
        return BoundaryEvent(
            segment_id_old=old,
            segment_id_new=new,
            scale=b.scale,
            posterior=float(b.posterior),
            sources=list(b.sources),
            t=int(t),
        )

    def _retime(self, ev: Any, t: int) -> Any:
        """Re-emit a real BoundaryEvent at tick t, preserving its payload."""
        if self.config.rewrite_segment_ids:
            old, new = self._next_segment_ids(str(getattr(ev, "scale", "")))
            return replace(ev, segment_id_old=old, segment_id_new=new, t=int(t))
        return replace(ev, t=int(t))

    # -- audit --------------------------------------------------------------- #

    def _episode_audit(self) -> Dict[str, Any]:
        true_train = self._true_train or BoundaryTrain(0, self._episode_index, 0)
        emitted = self._emitted_train or BoundaryTrain(0, self._episode_index, 0)
        donor = self._donor if self._donor is not None else true_train

        cfg = self.config
        n_true = true_train.count
        n_emit = emitted.count
        n_donor = donor.count

        # For donor modes the reference is the DONOR (that is what was
        # permuted); for jitter/off the reference is this episode's true train.
        ref = donor if cfg.requires_donor else true_train

        count_match = (n_emit == ref.count)
        iei_match = sorted(emitted.intervals()) == sorted(ref.intervals())
        post_match = _float_multiset_eq(emitted.posteriors(), ref.posteriors())
        scale_match = emitted.scale_counts() == ref.scale_counts()

        return {
            "episode_index": int(self._episode_index),
            "mode": cfg.mode,
            "n_steps": int(self._n_steps),
            "n_boundaries_true": int(n_true),
            "n_boundaries_donor": int(n_donor),
            "n_boundaries_emitted": int(n_emit),
            "donor_present": self._donor is not None,
            "count_match": bool(count_match),
            "iei_multiset_match": bool(iei_match),
            "posterior_multiset_match": bool(post_match),
            "scale_mix_match": bool(scale_match),
            "n_flushed_at_episode_end": int(len(self._pending)),
            # Alignment between the true and the emitted boundary indicator
            # trains. Near 0 is the point of the arm; near 1 means the arm did
            # not actually displace anything (e.g. a degenerate shift).
            "true_emitted_alignment": _indicator_corr(
                true_train.times(), emitted.times(), self._n_steps
            ),
        }

    def preservation_report(self) -> Dict[str, Any]:
        """Level-1 preservation audit: what the arm preserved BY CONSTRUCTION.

        `preserved_by_construction` is the gate. It says nothing about the
        environmental input statistics -- that is Level 2, and it is
        `input_statistics_divergence()`, which needs the paired intact arm.
        """
        eps = list(self._episodes)
        cfg = self.config
        n_ep = len(eps)

        def _all(key: str) -> bool:
            return bool(eps) and all(bool(e.get(key)) for e in eps)

        count_ok = _all("count_match")
        iei_ok = _all("iei_multiset_match")
        post_ok = _all("posterior_multiset_match")
        scale_ok = _all("scale_mix_match")

        if cfg.is_intact:
            preserved = True
        elif cfg.is_lesion:
            # A lesion does not claim preservation. Say so rather than emitting
            # a False that reads like a defect.
            preserved = False
        elif cfg.mode == "jitter":
            # Jitter preserves count and payload, not the IEI multiset. That is
            # the documented cost of the donor-free fallback.
            preserved = bool(count_ok and post_ok and scale_ok)
        else:
            preserved = bool(count_ok and iei_ok and post_ok and scale_ok)

        if self._donor_missing_episodes:
            preserved = False

        alignments = [
            float(e.get("true_emitted_alignment", 0.0))
            for e in eps
            if e.get("true_emitted_alignment") is not None
        ]
        mean_align = sum(alignments) / len(alignments) if alignments else 0.0

        return {
            "arm_id": cfg.arm_id(),
            "mode": cfg.mode,
            "is_primary": cfg.mode == PRIMARY_MODE,
            "is_lesion": bool(cfg.is_lesion),
            "is_intact": bool(cfg.is_intact),
            "n_episodes": int(n_ep),
            "count_match_all": bool(count_ok),
            "iei_multiset_match_all": bool(iei_ok),
            "posterior_multiset_match_all": bool(post_ok),
            "scale_mix_match_all": bool(scale_ok),
            "donor_missing_episodes": list(self._donor_missing_episodes),
            "preserved_by_construction": bool(preserved),
            "mean_true_emitted_alignment": float(mean_align),
            "n_boundaries_true_total": int(
                sum(int(e.get("n_boundaries_true", 0)) for e in eps)
            ),
            "n_boundaries_emitted_total": int(
                sum(int(e.get("n_boundaries_emitted", 0)) for e in eps)
            ),
            "episodes": eps,
            "note": (
                "Level 1 only: preservation AT THE INTERVENTION SITE. "
                "Environmental input statistics are Level 2 and require the "
                "paired intact arm via input_statistics_divergence()."
            ),
        }

    def recorded_trains(self) -> List[BoundaryTrain]:
        """Donor trains banked by an intact ('off') run, for a scrambled arm.

        Only meaningful for mode='off'. A scrambled arm's own true trains are
        NOT a valid donor for anything: they were produced under intervention,
        so yoking to them would compare an arm against itself. Returns [] for
        any non-intact mode rather than silently handing back a bad donor.
        """
        if not self.config.is_intact:
            return []
        return list(self._donor_bank)

    def donor_index(self) -> Dict[Tuple[int, int], BoundaryTrain]:
        """Donors keyed by (seed, episode_index) -- the yoking key.

        The scrambled arm looks up its donor by this exact pair, which is what
        makes the preservation guarantee auditable seed-for-seed and
        episode-for-episode rather than merely distributional.
        """
        return {
            (int(tr.seed), int(tr.episode_index)): tr
            for tr in self.recorded_trains()
        }


# --------------------------------------------------------------------------- #
# Behavioural-reach precondition                                               #
# --------------------------------------------------------------------------- #


def assert_behavioural_reach(agent: Any, strict: bool = True) -> Dict[str, Any]:
    """Verify the boundary stream actually reaches behaviour in this config.

    Without at least one live consumer the arm is INERT -- it trivially preserves
    input statistics because it cannot change anything, and it then tests only
    statistics computed on the boundary stream itself. That is not what Q-081
    asks, and an inert control arm is the same class of defect as an inert arm
    knob (see `inert_arm_knob.py`): a conjunctive claim quietly loses a conjunct.

    Returns the reach report. Raises when strict and no consumer is live.
    """
    hippo = getattr(agent, "hippocampal", None)
    cfg = getattr(hippo, "config", None) if hippo is not None else None

    def _f(name: str) -> bool:
        return bool(getattr(cfg, name, False)) if cfg is not None else False

    segmenter_on = _f("use_event_segmenter")
    consumers = {
        "use_invalidation_trigger": _f("use_invalidation_trigger"),
        "use_anchor_sets": _f("use_anchor_sets"),
        "use_per_region_vs": _f("use_per_region_vs"),
        "use_staleness_accumulator": _f("use_staleness_accumulator"),
    }
    live = [k for k, v in consumers.items() if v]
    report = {
        "use_event_segmenter": bool(segmenter_on),
        "consumers": consumers,
        "live_consumers": live,
        "has_behavioural_reach": bool(segmenter_on and live),
    }
    if strict and not report["has_behavioural_reach"]:
        raise RuntimeError(
            "Q-081 landmark-removal arm has NO behavioural reach: "
            "use_event_segmenter=" + str(segmenter_on)
            + ", live consumers=" + repr(live)
            + ". The arm would be inert and the result vacuous. Enable the "
            "MECH-287 / MECH-269 consumer path, or self-route "
            "substrate_not_ready."
        )
    return report


# --------------------------------------------------------------------------- #
# Level 2: environmental input statistics                                      #
# --------------------------------------------------------------------------- #


def input_statistics_divergence(
    intact: Dict[str, Any],
    scrambled: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Measure whether the arm left the environmental input statistics intact.

    This is the constraint the claim names and the one a closed loop cannot
    satisfy by construction: any arm with behavioural reach can change what the
    agent encounters, and an arm without behavioural reach is vacuous. So the
    honest posture is to MEASURE it and let a breach disqualify the arm rather
    than quietly contaminate the readout.

    Both arguments are per-arm summaries with any subset of these keys:

        state_visitation   Mapping[state_key, count]  -- where the agent was
        action_counts      Mapping[action, count]     -- what it did
        obs_channel_mean   Sequence[float]            -- per-channel obs mean
        obs_channel_std    Sequence[float]            -- per-channel obs SD
        harm_events        int      + n_steps int     -- harm rate
        reward_events      int      + n_steps int     -- reward rate
        episode_lengths    Sequence[int]

    Missing keys are skipped, not defaulted -- a metric that was not recorded
    must not silently pass.

    Verdict `input_statistics_preserved` is False when ANY recorded metric
    breaches its pre-registered threshold. On False the arm is CONFOUNDED: a
    statistic that vanished may have vanished because the agent was somewhere
    else, and "the landmark structure mattered" is then not separable from
    "the input distribution moved". Report it, do not tune it.
    """
    thr = dict(DEFAULT_INPUT_STAT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)

    metrics: Dict[str, float] = {}
    breaches: List[str] = []
    skipped: List[str] = []

    # -- state visitation --------------------------------------------------- #
    if "state_visitation" in intact and "state_visitation" in scrambled:
        js = _js_divergence_counts(
            intact["state_visitation"], scrambled["state_visitation"]
        )
        metrics["state_visitation_js"] = js
    else:
        skipped.append("state_visitation_js")

    # -- action distribution ------------------------------------------------ #
    if "action_counts" in intact and "action_counts" in scrambled:
        metrics["action_distribution_js"] = _js_divergence_counts(
            intact["action_counts"], scrambled["action_counts"]
        )
    else:
        skipped.append("action_distribution_js")

    # -- event rates -------------------------------------------------------- #
    for key, mkey in (("harm_events", "harm_rate_abs"), ("reward_events", "reward_rate_abs")):
        if (
            key in intact and key in scrambled
            and intact.get("n_steps") and scrambled.get("n_steps")
        ):
            r_i = float(intact[key]) / float(intact["n_steps"])
            r_s = float(scrambled[key]) / float(scrambled["n_steps"])
            metrics[mkey] = abs(r_i - r_s)
        else:
            skipped.append(mkey)

    # -- episode length ----------------------------------------------------- #
    if intact.get("episode_lengths") and scrambled.get("episode_lengths"):
        m_i = _mean(intact["episode_lengths"])
        m_s = _mean(scrambled["episode_lengths"])
        denom = abs(m_i) if abs(m_i) > 1e-12 else 1.0
        metrics["episode_length_rel"] = abs(m_i - m_s) / denom
    else:
        skipped.append("episode_length_rel")

    # -- observation channel marginals -------------------------------------- #
    if (
        intact.get("obs_channel_mean") is not None
        and scrambled.get("obs_channel_mean") is not None
    ):
        mu_i = list(intact["obs_channel_mean"])
        mu_s = list(scrambled["obs_channel_mean"])
        sd_i = list(intact.get("obs_channel_std") or [])
        worst = 0.0
        n = min(len(mu_i), len(mu_s))
        for k in range(n):
            s = float(sd_i[k]) if k < len(sd_i) else 0.0
            # Standardise on the INTACT arm's SD -- the reference distribution.
            # A channel with no variance in the intact arm cannot be
            # standardised; fall back to the raw absolute difference, which is
            # the conservative direction (it cannot hide a shift).
            d = abs(float(mu_i[k]) - float(mu_s[k]))
            worst = max(worst, d / s if s > 1e-9 else d)
        metrics["obs_channel_mean_std"] = worst
    else:
        skipped.append("obs_channel_mean_std")

    for name, val in metrics.items():
        limit = thr.get(name)
        if limit is not None and val > limit:
            breaches.append(name)

    return {
        "metrics": metrics,
        "thresholds": {k: v for k, v in thr.items() if k in metrics},
        "breaches": breaches,
        "not_measured": skipped,
        "input_statistics_preserved": (not breaches) and bool(metrics),
        "verdict_note": (
            "CONFOUNDED -- the arm moved the input distribution; a vanished "
            "statistic is not attributable to landmark removal."
            if breaches
            else (
                "No metric recorded; verdict cannot be asserted."
                if not metrics
                else "Input statistics preserved within pre-registered bounds."
            )
        ),
    }


# --------------------------------------------------------------------------- #
# Small numeric helpers (stdlib only; no torch/numpy dependency)               #
# --------------------------------------------------------------------------- #


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs)) / len(xs) if xs else 0.0


def _float_multiset_eq(a: Sequence[float], b: Sequence[float], tol: float = 1e-9) -> bool:
    a2, b2 = sorted(float(x) for x in a), sorted(float(x) for x in b)
    if len(a2) != len(b2):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a2, b2))


def _js_divergence_counts(a: Dict[Any, float], b: Dict[Any, float]) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two count maps."""
    keys = set(a) | set(b)
    ta = float(sum(float(v) for v in a.values()))
    tb = float(sum(float(v) for v in b.values()))
    if ta <= 0 or tb <= 0:
        return 0.0
    out = 0.0
    for k in keys:
        p = float(a.get(k, 0.0)) / ta
        q = float(b.get(k, 0.0)) / tb
        m = 0.5 * (p + q)
        if p > 0:
            out += 0.5 * p * math.log2(p / m)
        if q > 0:
            out += 0.5 * q * math.log2(q / m)
    # Clamp: accumulated float error can push a numerically-identical pair a few
    # ulps below zero or above one.
    return float(min(1.0, max(0.0, out)))


def _indicator_corr(times_a: Sequence[int], times_b: Sequence[int], n_steps: int) -> float:
    """Pearson correlation of two binary boundary-indicator trains.

    Near 0 means the arm displaced the landmarks (the point of the arm). Near 1
    means it did not -- a degenerate shift, or a mode that emitted where it
    fired. Returns 0.0 when either train is constant, which is the correct
    reading for "no alignment measurable" and never a spurious 1.0.
    """
    n = int(n_steps)
    if n <= 1:
        return 0.0
    sa, sb = set(int(t) for t in times_a), set(int(t) for t in times_b)
    xa = [1.0 if i in sa else 0.0 for i in range(n)]
    xb = [1.0 if i in sb else 0.0 for i in range(n)]
    ma, mb = _mean(xa), _mean(xb)
    num = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
    da = math.sqrt(sum((x - ma) ** 2 for x in xa))
    db = math.sqrt(sum((y - mb) ** 2 for y in xb))
    if da <= 1e-12 or db <= 1e-12:
        return 0.0
    return float(num / (da * db))
