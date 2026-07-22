"""
Contracts for the Q-081 STRUCTURE-DESTROYING (landmark-removal) arm.

The companion module to `test_q081_surrogate_null.py`. The surrogate destroys
cross-stream alignment in the ANALYSIS; this arm destroys landmark structure in
the SYSTEM. Neither substitutes for the other, and a statistic that survives this
arm was measuring the clock.

What is pinned here, and why each one is load-bearing:

  1. PRESERVATION BY CONSTRUCTION. The primary mode must preserve boundary count,
     the inter-event-interval multiset, the posterior multiset (which sets
     broadcast_strength downstream) and the scale mix EXACTLY. If it does not,
     a dead statistic could be dead because the drive changed, not because the
     alignment did -- the confound the arm exists to exclude.
  2. ALIGNMENT ACTUALLY DESTROYED. An arm that preserves everything but also
     changes nothing is an inert knob (see inert_arm_knob.py): the conjunctive
     claim silently loses a conjunct and the run still passes.
  3. THE LESION IS NEVER THE PRIMARY. `suppress` removes the drive as well as the
     alignment; it must not be able to masquerade as the discriminator.
  4. BEHAVIOURAL REACH IS ASSERTED, NOT ASSUMED. With no live consumer the arm is
     vacuous and trivially "preserves input statistics".
  5. THE INPUT-STATISTICS VERDICT CANNOT PASS VACUOUSLY. An unmeasured metric
     must not read as a cleared one.
"""

import math

import pytest

from experiments._lib.q081_landmark_removal import (
    MODES,
    PRIMARY_MODE,
    BoundaryTrain,
    LandmarkRemovalConfig,
    LandmarkScrambler,
    RecordedBoundary,
    assert_behavioural_reach,
    input_statistics_divergence,
)


# --------------------------------------------------------------------------- #
# Fakes: a segmenter/agent stand-in with the same surface the wrapper uses.     #
# --------------------------------------------------------------------------- #


class _FakeSegmenter:
    """Fires at a scripted set of ticks, with the real segmenter's surface."""

    slow_scale_name = "slow"
    scale_id_format = "{outer}.{inner}"

    def __init__(self, fire_at):
        # fire_at: {t: [(scale, posterior)]}
        self.fire_at = dict(fire_at)
        self._outer = 0
        self._inner = 0

    def step(self, latent_dict, pe_dict, t):
        from ree_core.hippocampal.event_segmenter import BoundaryEvent

        out = []
        for scale, posterior in self.fire_at.get(int(t), []):
            old = f"{self._outer}.{self._inner}"
            if scale == self.slow_scale_name:
                self._outer += 1
                self._inner = 0
            else:
                self._inner += 1
            new = f"{self._outer}.{self._inner}"
            out.append(
                BoundaryEvent(
                    segment_id_old=old,
                    segment_id_new=new,
                    scale=scale,
                    posterior=float(posterior),
                    sources=["fake"],
                    t=int(t),
                )
            )
        return out


class _FakeHippoConfig:
    def __init__(self, **flags):
        self.use_event_segmenter = flags.get("use_event_segmenter", True)
        self.use_invalidation_trigger = flags.get("use_invalidation_trigger", True)
        self.use_anchor_sets = flags.get("use_anchor_sets", True)
        self.use_per_region_vs = flags.get("use_per_region_vs", False)
        self.use_staleness_accumulator = flags.get("use_staleness_accumulator", False)


class _FakeHippo:
    def __init__(self, segmenter, **flags):
        self.event_segmenter = segmenter
        self.config = _FakeHippoConfig(**flags)


class _FakeAgent:
    def __init__(self, segmenter, **flags):
        self.hippocampal = _FakeHippo(segmenter, **flags)


N_STEPS = 200
FIRE_AT = {
    10: [("fast", 0.4)],
    23: [("fast", 0.55)],
    24: [("slow", 0.9)],
    61: [("fast", 0.3)],
    97: [("fast", 0.7)],
    98: [("slow", 0.85)],
    140: [("fast", 0.5)],
    177: [("fast", 0.6)],
}


def _run(mode, donor=None, seed=7, n_steps=N_STEPS, fire_at=None, **cfg_kw):
    """Drive one episode through the wrapper and return (scrambler, emitted)."""
    seg = _FakeSegmenter(fire_at if fire_at is not None else FIRE_AT)
    agent = _FakeAgent(seg)
    scr = LandmarkScrambler(LandmarkRemovalConfig(mode=mode, seed=seed, **cfg_kw))
    scr.attach(agent)
    scr.begin_episode(0, n_steps=n_steps, seed=seed, donor=donor)
    emitted = []
    for t in range(n_steps):
        emitted.extend(seg.step(latent_dict={}, pe_dict=None, t=t))
    scr.end_episode()
    scr.detach()
    return scr, emitted


def _intact_donor(seed=7, n_steps=N_STEPS, fire_at=None):
    scr, _ = _run("off", seed=seed, n_steps=n_steps, fire_at=fire_at)
    trains = scr.recorded_trains()
    assert len(trains) == 1
    return trains[0]


# --------------------------------------------------------------------------- #
# 1. Wrapper mechanics                                                         #
# --------------------------------------------------------------------------- #


def test_intact_mode_is_a_passthrough_and_banks_a_donor():
    scr, emitted = _run("off")
    assert [e.t for e in emitted] == sorted(FIRE_AT)
    donors = scr.recorded_trains()
    assert len(donors) == 1
    assert donors[0].times() == sorted(FIRE_AT)
    assert donors[0].count == sum(len(v) for v in FIRE_AT.values())


def test_detach_restores_the_original_step_function():
    seg = _FakeSegmenter(FIRE_AT)
    original = seg.step
    agent = _FakeAgent(seg)
    scr = LandmarkScrambler(LandmarkRemovalConfig(mode=PRIMARY_MODE))
    scr.attach(agent)
    assert seg.step is not original
    assert scr.attached
    scr.detach()
    # Bound methods are recreated per access, so compare the underlying function
    # -- and assert no instance-level shadow was left behind.
    assert seg.step.__func__ is original.__func__
    assert "step" not in vars(seg)
    assert not scr.attached


def test_double_attach_raises_rather_than_nesting_two_scramblers():
    # A nested scrambler would be invisible in the manifest: the emitted train
    # would be twice-permuted with no record of it.
    seg = _FakeSegmenter(FIRE_AT)
    agent = _FakeAgent(seg)
    scr = LandmarkScrambler(LandmarkRemovalConfig(mode=PRIMARY_MODE))
    scr.attach(agent)
    with pytest.raises(RuntimeError, match="already attached"):
        scr.attach(agent)
    scr.detach()


def test_attach_refuses_when_the_segmenter_is_absent():
    class _NoSeg:
        hippocampal = None

    scr = LandmarkScrambler(LandmarkRemovalConfig(mode=PRIMARY_MODE))
    with pytest.raises(RuntimeError, match="hippocampal"):
        scr.attach(_NoSeg())


def test_ticks_outside_an_episode_pass_through_unscrambled():
    # Scrambling an unaudited stretch of ticks would produce an intervention
    # that no preservation report covers.
    seg = _FakeSegmenter(FIRE_AT)
    agent = _FakeAgent(seg)
    scr = LandmarkScrambler(LandmarkRemovalConfig(mode="suppress"))
    scr.attach(agent)
    out = [e for t in range(N_STEPS) for e in seg.step({}, None, t)]
    scr.detach()
    assert len(out) == sum(len(v) for v in FIRE_AT.values())


# --------------------------------------------------------------------------- #
# 2. Preservation by construction -- the load-bearing guarantee                 #
# --------------------------------------------------------------------------- #


def test_primary_mode_preserves_count_iei_multiset_posteriors_and_scale_mix():
    donor = _intact_donor()
    scr, emitted = _run(PRIMARY_MODE, donor=donor)
    rep = scr.preservation_report()

    assert rep["is_primary"] is True
    assert rep["preserved_by_construction"] is True
    assert rep["count_match_all"] is True
    assert rep["iei_multiset_match_all"] is True
    assert rep["posterior_multiset_match_all"] is True
    assert rep["scale_mix_match_all"] is True
    assert len(emitted) == donor.count


def test_primary_mode_emits_the_same_posterior_multiset_verbatim():
    # broadcast_strength = posterior * gain, so this multiset IS the invalidation
    # drive. If it drifted, a dead statistic could be dead from a weaker drive.
    donor = _intact_donor()
    _, emitted = _run(PRIMARY_MODE, donor=donor)
    assert sorted(e.posterior for e in emitted) == pytest.approx(
        sorted(donor.posteriors())
    )


def test_permuted_intervals_cannot_run_off_the_end_of_the_episode():
    # A permutation of the interval multiset has the same cumulative sum, so the
    # last emission lands exactly on the donor's last boundary tick. This is why
    # the count is preserved EXACTLY rather than approximately -- no clipping.
    donor = _intact_donor()
    for seed in range(12):
        _, emitted = _run(PRIMARY_MODE, donor=donor, seed=seed)
        assert len(emitted) == donor.count
        assert max(e.t for e in emitted) == max(donor.times())


def test_circular_shift_preserves_count_and_interval_multiset_up_to_one_wrap():
    donor = _intact_donor()
    scr, emitted = _run("circular_shift", donor=donor, seed=3)
    rep = scr.preservation_report()
    assert len(emitted) == donor.count
    assert rep["count_match_all"] is True
    assert rep["posterior_multiset_match_all"] is True
    assert rep["scale_mix_match_all"] is True


def test_jitter_preserves_count_exactly_but_not_the_iei_multiset():
    # The documented cost of the donor-free fallback, pinned so it cannot be
    # mistaken for a defect -- or for an equivalent of the primary.
    scr, emitted = _run("jitter", seed=5, jitter_min_steps=5, jitter_max_steps=30)
    rep = scr.preservation_report()
    n_true = sum(len(v) for v in FIRE_AT.values())
    assert len(emitted) == n_true
    assert rep["count_match_all"] is True
    assert rep["posterior_multiset_match_all"] is True
    assert rep["iei_multiset_match_all"] is False
    assert rep["preserved_by_construction"] is True


def test_missing_donor_is_recorded_and_fails_preservation_rather_than_raising():
    # An expensive run must not die on one missing donor -- but it must not
    # silently claim preservation either.
    scr, emitted = _run(PRIMARY_MODE, donor=None)
    rep = scr.preservation_report()
    assert rep["donor_missing_episodes"] == [0]
    assert rep["preserved_by_construction"] is False
    assert emitted == []


# --------------------------------------------------------------------------- #
# 3. Alignment is actually destroyed -- the arm is not an inert knob            #
# --------------------------------------------------------------------------- #


def test_primary_mode_actually_displaces_the_landmarks():
    donor = _intact_donor()
    moved = 0
    for seed in range(10):
        _, emitted = _run(PRIMARY_MODE, donor=donor, seed=seed)
        if [e.t for e in emitted] != donor.times():
            moved += 1
    # Some permutations of a short train are near-identity; the arm must not be
    # an identity map in general.
    assert moved >= 8


def test_intact_arm_has_alignment_one_and_primary_arm_has_much_less():
    # The contrast IS the arm. If a mode preserved everything and moved nothing,
    # it would be an inert knob: the run passes while testing nothing.
    scr_off, _ = _run("off")
    assert scr_off.preservation_report()["mean_true_emitted_alignment"] == pytest.approx(
        1.0, abs=1e-9
    )

    donor = _intact_donor()
    aligns = []
    for seed in range(10):
        scr, _ = _run(PRIMARY_MODE, donor=donor, seed=seed)
        aligns.append(scr.preservation_report()["mean_true_emitted_alignment"])
    assert sum(aligns) / len(aligns) < 0.5


def test_emitted_segment_ids_are_self_consistent_and_monotonic():
    # Consumers key anchors on segment_id_new; the emitted stream must look like
    # a coherent segmentation that is merely misaligned, not like ids that jump.
    donor = _intact_donor()
    _, emitted = _run(PRIMARY_MODE, donor=donor, seed=1)
    for a, b in zip(emitted, emitted[1:]):
        assert a.segment_id_new == b.segment_id_old
    outers = [int(e.segment_id_new.split(".")[0]) for e in emitted]
    assert outers == sorted(outers)
    # A slow fire resets inner to 0; the fake train contains two of them.
    assert any(e.scale == "slow" for e in emitted)
    for e in emitted:
        if e.scale == "slow":
            assert e.segment_id_new.endswith(".0")


def test_rewrite_segment_ids_false_leaves_ids_untouched():
    donor = _intact_donor()
    _, emitted = _run(PRIMARY_MODE, donor=donor, seed=1, rewrite_segment_ids=False)
    assert all(e.segment_id_old == e.segment_id_new for e in emitted)


def test_scrambling_is_deterministic_under_a_fixed_seed():
    donor = _intact_donor()
    a = [(e.t, e.scale, e.posterior) for e in _run(PRIMARY_MODE, donor=donor, seed=11)[1]]
    b = [(e.t, e.scale, e.posterior) for e in _run(PRIMARY_MODE, donor=donor, seed=11)[1]]
    assert a == b


# --------------------------------------------------------------------------- #
# 4. The lesion must never be mistaken for the primary                          #
# --------------------------------------------------------------------------- #


def test_suppress_is_flagged_as_a_lesion_and_is_not_the_primary():
    scr, emitted = _run("suppress")
    rep = scr.preservation_report()
    assert emitted == []
    assert rep["is_lesion"] is True
    assert rep["is_primary"] is False
    # A lesion does not claim preservation: it removes the drive as well as the
    # alignment, so "preserved" would be a false reassurance.
    assert rep["preserved_by_construction"] is False


def test_only_one_mode_is_the_declared_primary():
    assert PRIMARY_MODE in MODES
    primaries = [
        m
        for m in MODES
        if LandmarkRemovalConfig(mode=m).arm_id() == "q081_landmark_" + PRIMARY_MODE
    ]
    assert primaries == [PRIMARY_MODE]
    assert LandmarkRemovalConfig(mode=PRIMARY_MODE).is_lesion is False
    assert LandmarkRemovalConfig(mode=PRIMARY_MODE).requires_donor is True


def test_a_scrambled_arm_never_offers_its_own_train_as_a_donor():
    # Yoking to a train produced under intervention would compare the arm with
    # itself and quietly destroy the matching guarantee.
    donor = _intact_donor()
    scr, _ = _run(PRIMARY_MODE, donor=donor)
    assert scr.recorded_trains() == []
    assert scr.donor_index() == {}


def test_donor_index_is_keyed_on_seed_and_episode():
    seg = _FakeSegmenter(FIRE_AT)
    agent = _FakeAgent(seg)
    scr = LandmarkScrambler(LandmarkRemovalConfig(mode="off"))
    scr.attach(agent)
    for ep in range(3):
        scr.begin_episode(ep, n_steps=N_STEPS, seed=42)
        for t in range(N_STEPS):
            seg.step({}, None, t)
        scr.end_episode()
    scr.detach()
    idx = scr.donor_index()
    assert set(idx) == {(42, 0), (42, 1), (42, 2)}


def test_unknown_mode_is_rejected_at_construction():
    with pytest.raises(ValueError, match="mode must be one of"):
        LandmarkRemovalConfig(mode="scramble_everything")


# --------------------------------------------------------------------------- #
# 5. Behavioural reach                                                          #
# --------------------------------------------------------------------------- #


def test_behavioural_reach_passes_when_a_consumer_is_live():
    agent = _FakeAgent(_FakeSegmenter(FIRE_AT))
    rep = assert_behavioural_reach(agent)
    assert rep["has_behavioural_reach"] is True
    assert "use_anchor_sets" in rep["live_consumers"]


def test_behavioural_reach_raises_when_the_arm_would_be_inert():
    # No consumer -> the arm cannot change anything -> it would trivially
    # "preserve input statistics" while testing nothing Q-081 asks about.
    agent = _FakeAgent(
        _FakeSegmenter(FIRE_AT),
        use_invalidation_trigger=False,
        use_anchor_sets=False,
        use_per_region_vs=False,
        use_staleness_accumulator=False,
    )
    with pytest.raises(RuntimeError, match="NO behavioural reach"):
        assert_behavioural_reach(agent)
    rep = assert_behavioural_reach(agent, strict=False)
    assert rep["has_behavioural_reach"] is False
    assert rep["live_consumers"] == []


def test_behavioural_reach_requires_the_segmenter_itself():
    agent = _FakeAgent(_FakeSegmenter(FIRE_AT), use_event_segmenter=False)
    with pytest.raises(RuntimeError, match="NO behavioural reach"):
        assert_behavioural_reach(agent)


# --------------------------------------------------------------------------- #
# 6. Input statistics -- the closed-loop constraint                             #
# --------------------------------------------------------------------------- #


def _arm_summary(states, actions, harm=5, reward=7, n_steps=1000, lengths=(100, 100)):
    return {
        "state_visitation": states,
        "action_counts": actions,
        "harm_events": harm,
        "reward_events": reward,
        "n_steps": n_steps,
        "episode_lengths": list(lengths),
        "obs_channel_mean": [0.5, 0.2],
        "obs_channel_std": [0.3, 0.3],
    }


def test_identical_arms_are_reported_as_preserved():
    s = _arm_summary({"a": 50, "b": 50}, {0: 30, 1: 70})
    out = input_statistics_divergence(s, dict(s))
    assert out["input_statistics_preserved"] is True
    assert out["breaches"] == []
    assert out["metrics"]["state_visitation_js"] == pytest.approx(0.0, abs=1e-9)


def test_a_grossly_shifted_visitation_distribution_is_flagged_confounded():
    # The failure this exists to catch: the statistic vanished because the agent
    # was somewhere else, not because landmark removal did anything.
    intact = _arm_summary({"a": 90, "b": 10}, {0: 50, 1: 50})
    scrambled = _arm_summary({"a": 10, "b": 90}, {0: 50, 1: 50})
    out = input_statistics_divergence(intact, scrambled)
    assert out["input_statistics_preserved"] is False
    assert "state_visitation_js" in out["breaches"]
    assert "CONFOUNDED" in out["verdict_note"]


def test_an_unmeasured_metric_cannot_pass_vacuously():
    # A metric that was never recorded must be listed as not-measured, never
    # counted as cleared.
    intact = {"n_steps": 100, "harm_events": 3}
    scrambled = {"n_steps": 100, "harm_events": 3}
    out = input_statistics_divergence(intact, scrambled)
    assert "state_visitation_js" in out["not_measured"]
    assert "action_distribution_js" in out["not_measured"]
    assert out["metrics"] == {"harm_rate_abs": pytest.approx(0.0)}
    assert out["input_statistics_preserved"] is True  # the one measured metric cleared


def test_no_metrics_at_all_yields_no_verdict_rather_than_a_pass():
    out = input_statistics_divergence({}, {})
    assert out["metrics"] == {}
    assert out["input_statistics_preserved"] is False
    assert "cannot be asserted" in out["verdict_note"]


def test_obs_channel_shift_is_standardised_on_the_intact_arms_sd():
    intact = _arm_summary({"a": 1}, {0: 1})
    scrambled = dict(_arm_summary({"a": 1}, {0: 1}))
    # 0.3 SD channel moved by 0.15 -> 0.5 SD, over the 0.25 threshold.
    scrambled["obs_channel_mean"] = [0.65, 0.2]
    out = input_statistics_divergence(intact, scrambled)
    assert out["metrics"]["obs_channel_mean_std"] == pytest.approx(0.5, abs=1e-9)
    assert "obs_channel_mean_std" in out["breaches"]


def test_zero_variance_channel_falls_back_to_the_raw_difference():
    # Cannot standardise on a zero SD; the fallback must be conservative (it
    # must not hide a shift by dividing by something tiny or by skipping).
    intact = _arm_summary({"a": 1}, {0: 1})
    intact["obs_channel_std"] = [0.0, 0.0]
    scrambled = dict(intact)
    scrambled["obs_channel_mean"] = [0.9, 0.2]
    out = input_statistics_divergence(intact, scrambled)
    assert out["metrics"]["obs_channel_mean_std"] == pytest.approx(0.4, abs=1e-9)
    assert "obs_channel_mean_std" in out["breaches"]


def test_thresholds_are_pre_registered_and_overridable_explicitly():
    intact = _arm_summary({"a": 90, "b": 10}, {0: 50, 1: 50})
    scrambled = _arm_summary({"a": 10, "b": 90}, {0: 50, 1: 50})
    strict = input_statistics_divergence(intact, scrambled)
    assert strict["input_statistics_preserved"] is False
    loose = input_statistics_divergence(
        intact, scrambled, thresholds={"state_visitation_js": 1.0}
    )
    assert "state_visitation_js" not in loose["breaches"]


def test_js_divergence_is_bounded_and_symmetric():
    a = {"x": 1.0}
    b = {"y": 1.0}
    out_ab = input_statistics_divergence(
        {"state_visitation": a, "n_steps": 1}, {"state_visitation": b, "n_steps": 1}
    )["metrics"]["state_visitation_js"]
    out_ba = input_statistics_divergence(
        {"state_visitation": b, "n_steps": 1}, {"state_visitation": a, "n_steps": 1}
    )["metrics"]["state_visitation_js"]
    assert out_ab == pytest.approx(1.0, abs=1e-9)
    assert out_ab == pytest.approx(out_ba, abs=1e-12)


# --------------------------------------------------------------------------- #
# 7. Round-tripping the donor bank                                              #
# --------------------------------------------------------------------------- #


def test_boundary_train_round_trips_through_json_shaped_dicts():
    # An intact arm must be bankable so a later scrambled arm can consume it
    # without re-running the intact arm.
    donor = _intact_donor()
    again = BoundaryTrain.from_dict(donor.as_dict())
    assert again.times() == donor.times()
    assert again.posteriors() == pytest.approx(donor.posteriors())
    assert again.scale_counts() == donor.scale_counts()
    assert again.intervals() == donor.intervals()


def test_intervals_are_a_complete_reparameterisation_of_the_times():
    # cumsum(intervals) == times is what guarantees a permutation lands inside
    # the episode; if this ever stopped holding the count guarantee would go too.
    donor = _intact_donor()
    cursor = 0
    rebuilt = []
    for d in donor.intervals():
        cursor += d
        rebuilt.append(cursor)
    assert rebuilt == donor.times()


def test_empty_donor_produces_an_empty_but_valid_arm():
    donor = BoundaryTrain(seed=1, episode_index=0, n_steps=50)
    scr, emitted = _run(PRIMARY_MODE, donor=donor, n_steps=50, fire_at={})
    assert emitted == []
    rep = scr.preservation_report()
    assert rep["preserved_by_construction"] is True
    assert rep["n_boundaries_emitted_total"] == 0
