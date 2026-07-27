"""ARC-071 policy_composition_via_repeated_grounding -- interface contracts.

Interface-level guarantees, NOT magnitude thresholds. The magnitudes (does the
accumulator fire often enough, does latency actually drop) belong to the queued
validation experiments, not here.

C1  config defaults + from_dims surfaces them + master OFF is bit-identical
C2  MECH-094 STRICT: no chunk can ever form from hypothesis_tag=True content
C3  MECH-322 carve-out is OFF by default even when chunking is ON
C4  MECH-322 requires ALL THREE conditions; each fails closed independently
C5  MECH-322 accelerated dissolution retires uncorroborated replay chunks
C6  MECH-323 formation requires repetition AND low variance AND the evaluative gate
C7  MECH-324 hysteresis: F_low < F_high, and formation-only leaves chunks uncrystallised
C8  R4 options structure + chunks-of-chunks depth cap
C9  proposal injection is off by default and additive when on
C10 MECH-324 dissolution is suppression-with-retention (Barnes 2005 / Bouton 2012)
"""

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.policy import (
    ChunkedPrimitive,
    ChunkLibrary,
    ChunkState,
    PolicyChunking,
    PolicyChunkingConfig,
)
from ree_core.utils.config import REEConfig


def _cfg(**kw):
    base = dict(use_policy_chunking=True, use_chunk_maintenance=True)
    base.update(kw)
    return PolicyChunkingConfig(**base)


def _run(pc, trials=60, good=(0, 1, 2, 3), bad=(3, 0, 3, 0)):
    """Drive a discriminating regime: `good` earns 1.0, `bad` earns 0.0."""
    for trial in range(trials):
        seq = good if trial % 2 == 0 else bad
        for a in seq:
            pc.record_step(a)
        pc.note_outcome(1.0 if trial % 2 == 0 else 0.0)
        pc.end_episode()


# ----------------------------------------------------------------------
# C1 -- config surface + OFF is bit-identical
# ----------------------------------------------------------------------
def test_c1_defaults_are_off_and_from_dims_forwards():
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_policy_chunking is False
    assert cfg.use_chunk_maintenance is False
    assert cfg.use_chunk_replay_origin_path is False
    assert cfg.use_chunk_proposal_injection is False
    # The hippocampal mirror must track the top-level knob, or the proposer
    # would never see it (the REEConfig.from_dims three-site hazard).
    assert cfg.hippocampal.use_chunk_proposal_injection is False

    agent = REEAgent(cfg)
    assert agent.policy_chunking is None
    assert agent.note_chunk_outcome(1.0) == []
    assert agent.get_chunking_state() == {}
    assert agent.note_chunk_replay_sequence([0, 1], 5.0) is None


def test_c1_from_dims_forwards_non_default_values():
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_chunking=True,
        chunk_min_repetitions=7,
        chunk_variance_low=0.11,
        use_chunk_proposal_injection=True,
    )
    assert cfg.chunk_min_repetitions == 7
    assert cfg.hippocampal.use_chunk_proposal_injection is True
    agent = REEAgent(cfg)
    assert agent.policy_chunking is not None
    assert agent.policy_chunking.config.min_repetitions == 7
    assert agent.policy_chunking.config.variance_low == pytest.approx(0.11)


# ----------------------------------------------------------------------
# C2 -- MECH-094 STRICT. The safety-critical contract.
# ----------------------------------------------------------------------
def test_c2_imagined_steps_can_never_form_a_chunk():
    """A hallucinated chunk would be catastrophic. No parameterisation permits it."""
    pc = PolicyChunking(_cfg(min_repetitions=2, crystallisation_min=1))
    for trial in range(80):
        for a in (0, 1, 2, 3):
            assert pc.record_step(a, hypothesis_tag=True) is False
        pc.note_outcome(1.0 if trial % 2 == 0 else 0.0)
        pc.end_episode()
    state = pc.get_state()
    assert state["chunk_acc_n_formed"] == 0
    assert state["chunk_acc_n_replay_formed"] == 0
    assert state["chunk_acc_n_steps"] == 0
    assert state["chunk_lib_size"] == 0
    # The refusals must be counted, not silently dropped.
    assert state["chunk_acc_n_simulation_skips"] == 80 * 4


def test_c2_mixed_stream_records_only_the_real_steps():
    pc = PolicyChunking(_cfg())
    pc.record_step(1, hypothesis_tag=False)
    pc.record_step(2, hypothesis_tag=True)
    pc.record_step(3, hypothesis_tag=False)
    assert pc.get_state()["chunk_acc_n_steps"] == 2
    assert pc.get_state()["chunk_acc_n_simulation_skips"] == 1


# ----------------------------------------------------------------------
# C3 / C4 -- MECH-322 carve-out
# ----------------------------------------------------------------------
def test_c3_carveout_off_by_default_even_when_chunking_on():
    pc = PolicyChunking(_cfg())
    _run(pc, trials=30)
    assert pc.config.use_chunk_replay_origin_path is False
    assert pc.note_replay_sequence([0, 1], value_tag=1e9, in_sleep_phase=True) is None
    assert pc.get_state()["chunk_acc_n_replay_formed"] == 0


@pytest.mark.parametrize(
    "value_tag,in_sleep,expect_none",
    [
        (1e9, False, True),   # (b) waking DMN -- refused
        (-1e9, True, True),   # (a) value below the high-positive bar -- refused
        (1e9, True, False),   # all conditions met -- permitted
    ],
)
def test_c4_carveout_conditions_each_fail_closed(value_tag, in_sleep, expect_none):
    pc = PolicyChunking(_cfg(use_chunk_replay_origin_path=True))
    _run(pc, trials=40)
    got = pc.note_replay_sequence([0, 1], value_tag=value_tag, in_sleep_phase=in_sleep)
    assert (got is None) is expect_none
    if got is not None:
        assert got.replay_origin is True


def test_c4_carveout_fails_closed_with_no_real_execution_history():
    """With no real outcomes the value bar is unreachable, so nothing can mint."""
    pc = PolicyChunking(_cfg(use_chunk_replay_origin_path=True))
    assert pc.note_replay_sequence([0, 1], value_tag=1e9, in_sleep_phase=True) is None


def test_c5_uncorroborated_replay_chunk_is_retired_on_deadline():
    pc = PolicyChunking(
        _cfg(use_chunk_replay_origin_path=True, replay_corroboration_episodes=3)
    )
    _run(pc, trials=40)
    chunk = pc.note_replay_sequence([0, 1], value_tag=1e9, in_sleep_phase=True)
    assert chunk is not None
    chunk.state = ChunkState.CRYSTALLISED
    for _ in range(3):
        pc.end_episode()
    assert pc.library.get([0, 1]).state is ChunkState.DISSOLVED
    assert pc.get_state()["chunk_lib_n_replay_deadline_dissolutions"] == 1


def test_c5_real_execution_clears_the_deadline():
    lib = ChunkLibrary(_cfg(replay_corroboration_episodes=3))
    chunk = ChunkedPrimitive(
        sequence=(0, 1), replay_origin=True, state=ChunkState.CRYSTALLISED
    )
    lib.register(chunk)
    lib.note_episode_end()
    assert chunk.episodes_since_corroboration == 1
    lib.note_real_execution((0, 1), outcome_variance=0.0)
    assert chunk.episodes_since_corroboration == 0


# ----------------------------------------------------------------------
# C6 -- MECH-323 joint formation condition
# ----------------------------------------------------------------------
def test_c6_accumulator_fires_on_a_repeating_rewarded_subsequence():
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60, crystallisation_min=2))
    _run(pc, trials=60)
    assert pc.get_state()["chunk_acc_n_formed"] > 0
    assert len(pc.selectable_chunks()) > 0


def test_c6_uniform_outcomes_form_nothing():
    """The evaluative gate is RELATIVE: with no outcome contrast, nothing forms."""
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60))
    for _ in range(60):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
        pc.note_outcome(1.0)
        pc.end_episode()
    assert pc.get_state()["chunk_acc_n_formed"] == 0


def test_c6_inconsistent_outcomes_form_nothing():
    """High outcome variance must block formation even with ample repetition."""
    pc = PolicyChunking(_cfg(min_repetitions=3, window_trials=60, variance_low=0.01))
    for trial in range(60):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
        pc.note_outcome(10.0 if trial % 2 == 0 else -10.0)
        pc.end_episode()
    assert pc.get_state()["chunk_acc_n_formed"] == 0


def test_c6_too_few_repetitions_form_nothing():
    pc = PolicyChunking(_cfg(min_repetitions=50, window_trials=60))
    _run(pc, trials=20)
    assert pc.get_state()["chunk_acc_n_formed"] == 0


# ----------------------------------------------------------------------
# C7 -- MECH-324 maintenance
# ----------------------------------------------------------------------
def test_c7_hysteresis_gap_is_enforced_by_config_validation():
    with pytest.raises(ValueError):
        PolicyChunkingConfig(variance_low=0.5, variance_high=0.2).validate()
    with pytest.raises(ValueError):
        PolicyChunkingConfig(variance_low=0.3, variance_high=0.3).validate()


def test_c7_formation_only_arm_never_crystallises():
    """ARM_1 (MECH-323 without MECH-324): chunks form but never crystallise.

    This is the registered dissociation -- Smith & Graybiel 2013's IL-disruption
    contrast. Without it the two operators would be untestable separately.
    """
    pc = PolicyChunking(_cfg(use_chunk_maintenance=False, min_repetitions=5))
    _run(pc, trials=60)
    state = pc.get_state()
    assert state["chunk_acc_n_formed"] > 0, "formation must still occur"
    assert state["chunk_lib_n_crystallised"] == 0, "maintenance off -> no crystallisation"
    assert pc.selectable_chunks() == []


def test_c7_dissolution_is_slower_than_formation_and_recoverable():
    lib = ChunkLibrary(_cfg(dissolve_trials=10, variance_high=0.45))
    chunk = ChunkedPrimitive(
        sequence=(0, 1), state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    lib.register(chunk)
    # Variance above F_high starts a SLOW decay, not an immediate removal.
    lib.tick_maintenance({(0, 1): 0.9})
    assert chunk.state is ChunkState.DISSOLVING
    assert 0.0 < chunk.selection_weight < 1.0
    # Recovery when variance falls back inside the band.
    lib.note_real_execution((0, 1), outcome_variance=0.1)
    assert chunk.state is ChunkState.CRYSTALLISED
    # Sustained high variance eventually dissolves it.
    chunk.state = ChunkState.DISSOLVING
    chunk.dissolving_trials = 0
    for _ in range(10):
        lib.tick_maintenance({(0, 1): 0.9})
    assert chunk.state is ChunkState.DISSOLVED
    assert chunk.selection_weight == 0.0


# ----------------------------------------------------------------------
# C8 -- options structure + recursion cap
# ----------------------------------------------------------------------
def test_c8_chunk_carries_sutton_options_fields():
    chunk = ChunkedPrimitive(sequence=(0, 1))
    assert hasattr(chunk, "initiation_set")
    assert hasattr(chunk, "termination_condition")
    assert chunk.termination_condition == "sequence_complete"
    assert chunk.replay_origin is False
    assert chunk.depth == 1
    assert "sequence" in chunk.as_dict()


def test_c8_depth_ladder_and_cap():
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60, max_depth=2))
    _run(pc, trials=60)
    depths = sorted({c.depth for c in pc.library.all_chunks()})
    assert depths, "expected at least one chunk"
    assert max(depths) <= 2, "max_depth must cap chunks-of-chunks recursion"


def test_c8_chunk_size_budget_is_respected():
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60, min_chunk_size=2, max_chunk_size=3))
    _run(pc, trials=60)
    for chunk in pc.library.all_chunks():
        assert 2 <= len(chunk.sequence) <= 3


def test_c8_library_size_is_bounded():
    pc = PolicyChunking(_cfg(min_repetitions=2, window_trials=200, max_library_size=3))
    _run(pc, trials=80)
    assert len(pc.library.all_chunks()) <= 3


# ----------------------------------------------------------------------
# C9 -- proposal injection
# ----------------------------------------------------------------------
def test_c9_injection_off_leaves_proposer_untouched():
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4, use_policy_chunking=True
    )
    agent = REEAgent(cfg)
    assert getattr(agent.hippocampal, "_chunk_source", None) is None
    assert agent.hippocampal._build_chunk_candidates(
        z_self=torch.zeros(1, cfg.e1.self_dim), z_world=torch.zeros(1, cfg.e1.world_dim)
    ) == []


def test_c9_injection_on_registers_the_chunk_source():
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
    )
    agent = REEAgent(cfg)
    assert agent.hippocampal._chunk_source is agent.policy_chunking


def test_c9_no_selectable_chunks_yields_no_candidates():
    """An empty library must not perturb the pool even with injection on."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
    )
    agent = REEAgent(cfg)
    assert agent.hippocampal._build_chunk_candidates(
        z_self=torch.zeros(1, cfg.e1.self_dim), z_world=torch.zeros(1, cfg.e1.world_dim)
    ) == []


# ----------------------------------------------------------------------
# C10 -- MECH-324 dissolution is SUPPRESSION WITH RETENTION, not erasure.
#
# Barnes et al. 2005 (Nature 10.1038/nature04053): striatal ensemble patterns
# are "successively formed, reversed and then re-emerged", and "regaining a
# habit can occur quickly, with even one or a few exposures".
# Bouton, Winterbauer & Todd 2012 (10.1016/j.beproc.2012.03.004), INSTRUMENTAL
# extinction: it "weakens behavior without erasing the original learning".
#
# Rapid reacquisition is the one of the three relapse effects the substrate
# implements; renewal (context-gated dissolution) and resurgence (dissolution
# state shared across competitors) are unbuilt by design -- see the module
# docstring. These contracts pin the interface, not the magnitude.
# ----------------------------------------------------------------------
def _dissolve(pc, chunk):
    """Drive a chunk all the way to DISSOLVED through the real decay path."""
    chunk.state = ChunkState.DISSOLVING
    chunk.dissolving_trials = 0
    for _ in range(pc.config.dissolve_trials):
        pc.library.tick_maintenance({chunk.key: 0.99})
    assert chunk.state is ChunkState.DISSOLVED
    return chunk


def _cfg_retention(**kw):
    base = dict(
        min_repetitions=20,
        window_trials=60,
        crystallisation_min=2,
        dissolve_trials=3,
        use_chunk_dissolution_retention=True,
    )
    base.update(kw)
    return _cfg(**base)


def test_c10_retention_is_off_by_default_and_requires_maintenance():
    """Default OFF, and the dependency is LOUD rather than silently inert."""
    assert PolicyChunkingConfig().use_chunk_dissolution_retention is False
    assert PolicyChunkingConfig().reacquisition_repetition_factor == pytest.approx(0.25)
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_chunk_dissolution_retention is False
    assert cfg.chunk_reacquisition_repetition_factor == pytest.approx(0.25)

    # With maintenance off nothing ever dissolves, so retention could never
    # fire. That must RAISE, not run as a flag that reads enabled in a manifest
    # while its consumer never runs.
    with pytest.raises(ValueError) as excinfo:
        PolicyChunkingConfig(
            use_chunk_maintenance=False, use_chunk_dissolution_retention=True
        ).validate()
    assert "use_chunk_maintenance" in str(excinfo.value)


def test_c10_from_dims_forwards_retention_to_the_operator():
    """The REEConfig three-site hazard: field + signature + assignment + agent."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_chunking=True,
        use_chunk_maintenance=True,
        use_chunk_dissolution_retention=True,
        chunk_reacquisition_repetition_factor=0.5,
    )
    agent = REEAgent(cfg)
    assert agent.policy_chunking.config.use_chunk_dissolution_retention is True
    assert agent.policy_chunking.config.reacquisition_repetition_factor == pytest.approx(0.5)


def test_c10_reacquisition_factor_must_scale_down():
    """A factor above 1 would invert the acquisition/reacquisition asymmetry."""
    with pytest.raises(ValueError):
        PolicyChunkingConfig(reacquisition_repetition_factor=1.5).validate()
    with pytest.raises(ValueError):
        PolicyChunkingConfig(reacquisition_repetition_factor=0.0).validate()


@pytest.mark.parametrize(
    "r_min,factor,expected",
    [(20, 0.25, 5), (20, 1.0, 20), (10, 0.3, 3), (3, 0.25, 1), (1, 0.25, 1)],
)
def test_c10_reacquisition_bar_is_a_fraction_of_r_min(r_min, factor, expected):
    cfg = PolicyChunkingConfig(
        min_repetitions=r_min, window_trials=100, reacquisition_repetition_factor=factor
    )
    assert cfg.reacquisition_min_repetitions == expected
    assert cfg.reacquisition_min_repetitions <= cfg.min_repetitions


def test_c10_off_dissolved_is_an_absorbing_tombstone():
    """The uncorrected behaviour, pinned so the OFF arm stays the honest null.

    Worth stating plainly because it is worse than erasure, not better: the
    chunk is retained in the library dict, which BLOCKS the formation pass from
    ever re-minting the sequence, and nothing else can revive it. So an
    arbitrarily long stretch of the same perfectly consistent, above-baseline
    regime leaves it DISSOLVED.
    """
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60, crystallisation_min=2,
                             dissolve_trials=3))
    _run(pc, trials=60)
    chunk = _dissolve(pc, pc.library.all_chunks()[0])
    assert chunk.n_dissolutions == 1

    _run(pc, trials=200)
    assert pc.library.get(chunk.key) is chunk, "retained in the audit trail"
    assert chunk.state is ChunkState.DISSOLVED, "and never comes back"
    assert chunk.n_reacquisitions == 0
    assert pc.library.dormant_chunks() == [], "nothing is dormant with the flag off"
    assert pc.get_state()["chunk_lib_n_reacquisitions"] == 0


def test_c10_on_a_dissolved_chunk_reforms_below_r_min():
    """Rapid reacquisition: the reduced bar is what brings the chunk back.

    R_min is 20 and the reduced bar is 5. The chunk re-forms, and it does so on
    materially fewer post-dissolution repetitions than R_min -- which is the
    whole claim, and the measurement a validation experiment would make.
    """
    pc = PolicyChunking(_cfg_retention())
    _run(pc, trials=60)
    chunk = _dissolve(pc, pc.library.get((0, 1, 2, 3)) or pc.library.all_chunks()[0])
    assert pc.library.dormant_chunks(), "DISSOLVED must be DORMANT when retention is on"

    bar = pc.config.reacquisition_min_repetitions
    assert bar < pc.config.min_repetitions

    # Execute the chunk on every trial and count trials-to-re-formation. One
    # execution per trial, so the trial index IS the repetition count: the bar
    # is cleared at 5, where original acquisition would have needed R_min = 20.
    reformed_after = None
    for t in range(60):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
        pc.note_outcome(1.0 if t % 2 == 0 else 0.0)
        pc.end_episode()
        if chunk.state is not ChunkState.DISSOLVED:
            reformed_after = t + 1
            break

    assert reformed_after is not None, "a dormant chunk must be able to re-form"
    assert reformed_after == bar, (
        f"re-formation took {reformed_after} repetitions, expected the reduced "
        f"bar {bar}"
    )
    assert reformed_after < pc.config.min_repetitions, (
        "rapid reacquisition: re-forming a dormant chunk must need MATERIALLY "
        f"fewer than R_min={pc.config.min_repetitions} repetitions"
    )
    assert chunk.state is ChunkState.FORMING, (
        "revival returns a chunk to FORMING -- the C_min crystallisation counter "
        "is a separate sub-mechanism and must run again from zero"
    )
    assert chunk.crystallisation_counter == 0
    assert chunk.n_reacquisitions == 1
    assert chunk.n_dissolutions == 1
    assert pc.get_state()["chunk_lib_n_reacquisitions"] == 1


def test_c10_reacquisition_still_requires_consistency_and_contrast():
    """Retention lowers the price of coming back; it does not waive the evidence.

    The other two MECH-323 gates are applied unchanged, so a dormant chunk
    re-executed into an INCONSISTENT regime accrues repetitions past the bar and
    still does not re-form.
    """
    pc = PolicyChunking(_cfg_retention(variance_low=0.01))
    _run(pc, trials=60)
    chunk = _dissolve(pc, pc.library.all_chunks()[0])

    for t in range(60):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
        pc.note_outcome(10.0 if t % 2 == 0 else -10.0)  # high variance
        pc.end_episode()

    assert chunk.reacquisition_repetitions > pc.config.reacquisition_min_repetitions, (
        "the repetition bar must have been cleared, so the refusal is the "
        "variance gate and not simply too few repetitions"
    )
    assert chunk.state is ChunkState.DISSOLVED
    assert chunk.n_reacquisitions == 0


def test_c10_replay_origin_chunks_are_never_revived():
    """MECH-322 fails closed: a chunk retired for want of real corroboration
    must not return by a REDUCED threshold."""
    lib = ChunkLibrary(_cfg_retention())
    chunk = ChunkedPrimitive(
        sequence=(0, 1), replay_origin=True, state=ChunkState.DISSOLVED
    )
    lib.register(chunk)
    assert chunk.is_dormant is False
    assert lib.dormant_chunks() == []
    assert lib.revive((0, 1)) is False
    assert chunk.state is ChunkState.DISSOLVED
    assert lib.get_state()["chunk_lib_n_reacquisition_refusals"] == 1


def test_c10_revive_fails_closed_on_every_refusal_path():
    lib_off = ChunkLibrary(_cfg(dissolve_trials=3))
    live = ChunkedPrimitive(sequence=(0, 1), state=ChunkState.DISSOLVED)
    lib_off.register(live)
    assert lib_off.revive((0, 1)) is False, "retention off -> no revival at any setting"

    lib_on = ChunkLibrary(_cfg_retention())
    crystallised = ChunkedPrimitive(sequence=(2, 3), state=ChunkState.CRYSTALLISED)
    lib_on.register(crystallised)
    assert lib_on.revive((2, 3)) is False, "not DISSOLVED -> nothing to revive"
    assert lib_on.revive((9, 9)) is False, "absent sequence -> refused"
    assert crystallised.state is ChunkState.CRYSTALLISED
    # Refusals are counted, never silently dropped.
    assert lib_on.get_state()["chunk_lib_n_reacquisition_refusals"] == 2


def test_c10_dormant_chunks_are_still_suppressed():
    """A dormant chunk is SUPPRESSED: retained, but out of the proposal pool."""
    pc = PolicyChunking(_cfg_retention())
    _run(pc, trials=60)
    chunk = _dissolve(pc, pc.library.all_chunks()[0])
    assert chunk in pc.library.dormant_chunks()
    assert chunk.is_selectable is False
    assert chunk not in pc.selectable_chunks()
    assert chunk.selection_weight == 0.0
