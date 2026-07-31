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
C11 MECH-323 chunk SIZE is a growable ceiling derived from the deliberation budget
C12 ARC-071 chunk DEPTH is likewise derived, and bounded by the size ceiling
C13 MECH-323 credit rule: all-position crediting, and the two guards that stop
    it becoming a spurious PASS (per-outcome dedup, coupled corroboration)
"""

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.policy import (
    ChunkAccumulator,
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


def _dissolve_via_real_contamination(pc, seq, max_trials=30):
    """Drive DISSOLVED through the REAL state machine, contaminating the
    accumulator's tally for `seq` exactly as a real high-variance regime
    would (unlike `_dissolve()`, which forces the state directly and never
    touches the tally -- see V3-EXQ-829 / the MECH-324 reacquisition-window
    isolation design doc for why that shortcut hid the bug this fixes)."""
    chunk = pc.library.get(seq)
    for t in range(max_trials):
        for a in seq:
            pc.record_step(a)
        pc.note_outcome(10.0 if t % 2 == 0 else -10.0)
        pc.end_episode()
        if chunk.state is ChunkState.DISSOLVED:
            return t + 1
    return None


def test_c10_reacquisition_window_isolation_off_by_default_and_requires_retention():
    """Default OFF, and the dependency is LOUD rather than silently inert."""
    assert PolicyChunkingConfig().use_reacquisition_window_isolation is False
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_reacquisition_window_isolation is False

    with pytest.raises(ValueError) as excinfo:
        PolicyChunkingConfig(
            use_chunk_maintenance=True,
            use_chunk_dissolution_retention=False,
            use_reacquisition_window_isolation=True,
        ).validate()
    assert "use_chunk_dissolution_retention" in str(excinfo.value)


def test_c10_from_dims_forwards_window_isolation_to_the_operator():
    """The REEConfig three-site hazard: field + signature + assignment + agent."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_chunking=True,
        use_chunk_maintenance=True,
        use_chunk_dissolution_retention=True,
        use_reacquisition_window_isolation=True,
    )
    agent = REEAgent(cfg)
    assert agent.policy_chunking.config.use_reacquisition_window_isolation is True


def test_c10_window_isolation_off_reproduces_the_v3_exq_829_flat_signature():
    """The confirmed bug, pinned so the OFF arm stays the honest (buggy) null.

    Dissolving via REAL high-variance execution of the target sequence (not
    the `_dissolve()` shortcut) contaminates accumulator._tally[target] with
    the dissolution episode's own outcomes. With the isolation flag OFF, the
    revival gate reads that same contaminated, whole-lifetime tally, so
    clearing the variance gate takes on the order of window_trials real
    target executions -- NOT the (much lower) reacquisition bar. This is
    exactly V3-EXQ-829's measured median_r_reacq/window_trials ~ 0.9 signature,
    reproduced deterministically here as reformed-after == window_trials
    rather than == bar.
    """
    cfg = _cfg_retention(
        min_repetitions=20,
        window_trials=20,
        dissolve_trials=20,
        reacquisition_repetition_factor=0.15,  # bar = ceil(20 * 0.15) = 3
        use_reacquisition_window_isolation=False,
    )
    pc = PolicyChunking(cfg)
    _run(pc, trials=80)
    target, filler = (0, 1, 2, 3), (3, 0, 3, 0)
    chunk = pc.library.get(target)
    assert _dissolve_via_real_contamination(pc, target) is not None
    bar = pc.config.reacquisition_min_repetitions
    assert bar < pc.config.window_trials, "the test must set up a real gap to measure"

    # Re-present the target regime interleaved with a filler arm (mirrors
    # _run()'s own good/bad interleaving) so the population baseline stays
    # anchored below the target's reward instead of chasing it to a
    # mu-equals-baseline deadlock -- an artifact of a single-sequence-only
    # loop, not a substrate property.
    target_reps = 0
    reformed_after = None
    for t in range(120):
        seq = target if t % 2 == 0 else filler
        for a in seq:
            pc.record_step(a)
        pc.note_outcome(1.0 if seq is target else 0.0)
        pc.end_episode()
        if seq is target:
            target_reps += 1
        if chunk.state is not ChunkState.DISSOLVED:
            reformed_after = target_reps
            break

    assert reformed_after is not None, "must eventually reform once the window clears"
    assert reformed_after > bar, (
        "OFF must NOT reform at the reduced bar -- it should be bound by "
        "window_trials instead, which is the bug this fix corrects"
    )
    assert reformed_after == pc.config.window_trials, (
        f"reformed after {reformed_after} target repetitions, expected exactly "
        f"window_trials={pc.config.window_trials} (the contaminated window "
        "must fully flush before the variance gate can clear)"
    )


def test_c10_window_isolation_on_reforms_at_the_reduced_bar():
    """The fix: with isolation ON, reacquisition is bound by `bar`, not `W`.

    Identical setup to the OFF test above (same contamination path, same
    config apart from the flag) -- only the flag changes the outcome, which
    is the point: this is a data-flow fix, not a threshold recalibration.
    """
    cfg = _cfg_retention(
        min_repetitions=20,
        window_trials=20,
        dissolve_trials=20,
        reacquisition_repetition_factor=0.15,  # bar = ceil(20 * 0.15) = 3
        use_reacquisition_window_isolation=True,
    )
    pc = PolicyChunking(cfg)
    _run(pc, trials=80)
    target, filler = (0, 1, 2, 3), (3, 0, 3, 0)
    chunk = pc.library.get(target)
    assert _dissolve_via_real_contamination(pc, target) is not None
    bar = pc.config.reacquisition_min_repetitions

    target_reps = 0
    reformed_after = None
    for t in range(120):
        seq = target if t % 2 == 0 else filler
        for a in seq:
            pc.record_step(a)
        pc.note_outcome(1.0 if seq is target else 0.0)
        pc.end_episode()
        if seq is target:
            target_reps += 1
        if chunk.state is not ChunkState.DISSOLVED:
            reformed_after = target_reps
            break

    assert reformed_after == bar, (
        f"reformed after {reformed_after} target repetitions, expected exactly "
        f"the reduced bar {bar} -- rapid reacquisition should now be governed "
        "by reacquisition_repetition_factor, not window_trials"
    )
    assert chunk.reacquisition_outcomes == [], (
        "revive() must reset the isolated window (symmetric with "
        "reacquisition_repetitions) so a later re-dissolution starts clean"
    )


def test_c10_window_isolation_numerical_floor_refuses_a_single_sample():
    """len(window) < 2 is a stability floor, not a second bar.

    _variance() is definitionally 0.0 below n=2, which would let one
    post-dissolution sample trivially clear the variance gate at bar==1
    settings with zero evidence of consistency. One sample must be refused;
    two (still consistent, still above baseline) must be judged and revive.
    """
    cfg = _cfg_retention(
        min_repetitions=4,
        window_trials=20,
        dissolve_trials=5,
        reacquisition_repetition_factor=0.25,  # bar = ceil(4 * 0.25) = 1
        use_reacquisition_window_isolation=True,
    )
    pc = PolicyChunking(cfg)
    _run(pc, trials=40)
    target = (0, 1, 2, 3)
    chunk = pc.library.get(target)
    assert _dissolve_via_real_contamination(pc, target) is not None
    assert pc.config.reacquisition_min_repetitions == 1

    for a in target:
        pc.record_step(a)
    pc.note_outcome(1.0)
    pc.end_episode()
    assert chunk.reacquisition_repetitions == 1
    assert len(chunk.reacquisition_outcomes) == 1
    assert chunk.state is ChunkState.DISSOLVED, (
        "bar is cleared (1 >= 1) but len(window) < 2 must still refuse"
    )

    for a in target:
        pc.record_step(a)
    pc.note_outcome(1.0)
    pc.end_episode()
    assert chunk.state is not ChunkState.DISSOLVED, (
        "len(window) == 2 must be judgeable and revive"
    )


# ----------------------------------------------------------------------
# C11 -- MECH-323 growable chunk-size ceiling (Ramkumar 2016 / Bo 2009)
#
# The budget 2-5 is an INITIAL budget, not a lifetime cap. These contracts pin
# the four things that make that safe: the derivation reproduces the inherited
# constant at today's deliberation budget, growth is licensed by realised
# marginal return (not by practice), the brake actually brakes, and OFF is
# bit-identical.
# ----------------------------------------------------------------------
def _cfg_ceiling(**kw):
    base = dict(use_growable_chunk_ceiling=True, chunk_deliberation_horizon=60,
                min_repetitions=3)
    base.update(kw)
    return _cfg(**base)


def _seed_ceiling_tally(acc, whole_outcomes, sub_outcomes):
    """Seed the tally at the current ceiling with explicit sub-sequence means."""
    ceiling = acc.effective_max_chunk_size
    key = tuple(range(1, ceiling + 1))
    acc._tally[key] = list(whole_outcomes)
    acc._tally[key[1:]] = list(sub_outcomes)
    acc._tally[key[:-1]] = list(sub_outcomes)
    return key


def test_c11_ceiling_is_off_by_default_and_pinned_when_off():
    acc = ChunkAccumulator(_cfg())
    assert acc.config.use_growable_chunk_ceiling is False
    assert acc.effective_max_chunk_size == acc.config.max_chunk_size
    _seed_ceiling_tally(acc, [1.0] * 8, [0.0] * 8)
    assert acc.consider_ceiling_growth() is False
    assert acc.effective_max_chunk_size == 5


def test_c11_derivation_reproduces_the_inherited_budget_at_reeds_real_horizon():
    """THE ANCHOR. A built agent must still derive exactly 5.

    Regression guard against silently raising the constant. Ramkumar 2016
    licenses no replacement number, so an agent at today's deliberation budget
    must be left exactly where it was; only a LARGER budget may derive more.
    Note this reads the real from_dims horizon (30), NOT the HippocampalConfig
    dataclass default (10) that no built agent actually uses.
    """
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4,
                              use_policy_chunking=True, use_growable_chunk_ceiling=True)
    agent = REEAgent(cfg)
    pcfg = agent.policy_chunking.config
    assert pcfg.chunk_deliberation_horizon == cfg.hippocampal.horizon == 30
    assert pcfg.derived_chunk_ceiling == 5


@pytest.mark.parametrize("horizon,expected", [(30, 5), (60, 10), (90, 12), (300, 12)])
def test_c11_ceiling_scales_with_the_deliberation_budget(horizon, expected):
    """Consequence (ii): different rollout budgets -> different chunk sizes."""
    assert PolicyChunkingConfig(
        chunk_deliberation_horizon=horizon).derived_chunk_ceiling == expected


def test_c11_growth_requires_a_realised_marginal_return():
    acc = ChunkAccumulator(_cfg_ceiling())
    # Whole predicts perfectly; BOTH one-shorter contexts are aliased.
    _seed_ceiling_tally(acc, [1.0] * 8, [1.0, 0.0] * 4)
    assert acc.marginal_return_at_ceiling() == pytest.approx(0.5)
    assert acc.consider_ceiling_growth() is True
    assert acc.effective_max_chunk_size == 6


def test_c11_no_growth_when_a_shorter_context_already_predicts():
    acc = ChunkAccumulator(_cfg_ceiling())
    _seed_ceiling_tally(acc, [1.0] * 8, [1.0] * 8)
    assert acc.marginal_return_at_ceiling() == pytest.approx(0.0)
    assert acc.consider_ceiling_growth() is False
    assert acc.effective_max_chunk_size == 5


def test_c11_no_evidence_is_not_a_gain_of_zero():
    """gain=None must refuse even at a zero threshold.

    Distinct from a measured gain of 0.0: an unattested ceiling would otherwise
    clear a >= 0.0 bar and grow on no evidence at all.
    """
    acc = ChunkAccumulator(_cfg_ceiling(chunk_ceiling_returns_threshold=0.0))
    assert acc.marginal_return_at_ceiling() is None
    assert acc.consider_ceiling_growth() is False


def test_c11_brake_plateaus_below_the_derived_maximum():
    """Guardrail 2. Growth must stop when returns flatten, NOT at the cap.

    This is the contract that forbids the monotonic accumulator: with headroom
    to 10 available, flat returns must leave the ceiling at 5.
    """
    acc = ChunkAccumulator(_cfg_ceiling())
    assert acc.config.derived_chunk_ceiling == 10
    for _ in range(30):
        _seed_ceiling_tally(acc, [1.0] * 8, [1.0] * 8)
        if not acc.consider_ceiling_growth():
            break
    assert acc.effective_max_chunk_size == 5
    assert acc._n_ceiling_growths == 0


def test_c11_growth_never_exceeds_the_derived_bound():
    acc = ChunkAccumulator(_cfg_ceiling(chunk_ceiling_returns_threshold=0.0))
    for _ in range(50):
        _seed_ceiling_tally(acc, [1.0] * 8, [0.0] * 8)
        acc.consider_ceiling_growth()
    assert acc.effective_max_chunk_size == acc.config.derived_chunk_ceiling == 10


def test_c11_growth_is_decoupled_from_the_repetition_tally():
    """Bo 2009: chunk SIZE and formation RATE are separable.

    Unbounded repetition with zero marginal return must not grow the ceiling.
    Pins that the growth rule reads outcome gain, never practice volume -- the
    coupling a naive practice-driven accumulator would reintroduce.
    """
    acc = ChunkAccumulator(_cfg_ceiling())
    _seed_ceiling_tally(acc, [1.0] * 500, [1.0] * 500)
    assert acc.consider_ceiling_growth() is False
    assert acc.effective_max_chunk_size == 5


def test_c11_ceiling_grows_end_to_end_so_the_flag_is_not_inert():
    """(9,1,2,3,4) is good; both 4-element contexts it contains are aliased."""
    branches = [([9, 1, 2, 3, 4], 1.0), ([8, 1, 2, 3, 4], 0.0), ([9, 1, 2, 3, 7], 0.0)]

    def drive(on):
        pc = PolicyChunking(_cfg(min_repetitions=5, use_growable_chunk_ceiling=on,
                                 chunk_deliberation_horizon=60))
        for ep in range(600):
            seq, out = branches[ep % 3]
            for a in seq:
                pc.record_step(a)
            pc.note_outcome(out)
            pc.end_episode()
        return pc.accumulator

    off, on = drive(False), drive(True)
    assert off.effective_max_chunk_size == 5 and off._n_ceiling_growths == 0
    assert on.effective_max_chunk_size > 5 and on._n_ceiling_growths >= 1
    # ...and still plateaus well short of the derived bound of 10.
    assert on.effective_max_chunk_size < on.config.derived_chunk_ceiling


def test_c11_reset_returns_the_ceiling_to_the_initial_budget():
    acc = ChunkAccumulator(_cfg_ceiling())
    _seed_ceiling_tally(acc, [1.0] * 8, [1.0, 0.0] * 4)
    assert acc.consider_ceiling_growth() is True
    acc.reset()
    assert acc.effective_max_chunk_size == acc.config.max_chunk_size
    assert acc._n_ceiling_growths == 0


@pytest.mark.parametrize("kw", [
    dict(chunk_ceiling_budget_fraction=0.0),
    dict(chunk_ceiling_budget_fraction=1.5),
    dict(chunk_ceiling_returns_threshold=-0.1),
    dict(chunk_ceiling_hard_max=3),
    dict(chunk_deliberation_horizon=0),
])
def test_c11_incoherent_ceiling_config_is_refused(kw):
    with pytest.raises(ValueError):
        PolicyChunkingConfig(**kw).validate()


def test_c11_from_dims_forwards_every_ceiling_knob():
    """All THREE wiring sites, plus agent.py's mapping.

    from_dims silently swallows unknown kwargs, so a knob wired at only two
    sites is unreachable with NO error -- this asserts the round-trip rather
    than the signature.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4,
        use_policy_chunking=True, use_growable_chunk_ceiling=True,
        chunk_deliberation_horizon=16, chunk_ceiling_budget_fraction=0.75,
        chunk_ceiling_returns_threshold=0.2, chunk_ceiling_hard_max=9)
    assert cfg.use_growable_chunk_ceiling is True
    assert cfg.chunk_deliberation_horizon == 16
    assert cfg.chunk_ceiling_budget_fraction == 0.75
    assert cfg.chunk_ceiling_returns_threshold == 0.2
    assert cfg.chunk_ceiling_hard_max == 9
    pcfg = REEAgent(cfg).policy_chunking.config
    assert pcfg.use_growable_chunk_ceiling is True
    assert pcfg.chunk_deliberation_horizon == 16  # explicit override beats the mirror
    assert pcfg.chunk_ceiling_hard_max == 9


def test_c11_sentinel_horizon_mirrors_the_hippocampal_budget():
    """0 = mirror the real budget; the override must stay reachable."""
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4,
                              use_policy_chunking=True, use_growable_chunk_ceiling=True)
    assert cfg.chunk_deliberation_horizon == 0
    assert REEAgent(cfg).policy_chunking.config.chunk_deliberation_horizon == 30


# ----------------------------------------------------------------------
# C12 -- ARC-071 growable chunk-DEPTH ceiling (Solway 2014)
#
# The R4/R3 cap of 3 was the other fiat constant on the compute/efficiency
# trade-off. These contracts pin the same four things C11 pins for size --
# the derivation reproduces the inherited constant at today's budget, growth is
# licensed by realised marginal return, the brake brakes, OFF is bit-identical
# -- plus the one that is specific to depth: growth stops at what the SIZE
# ceiling makes structurally reachable, so the ceiling can never be raised into
# inertness (the defect the MECH-321 scoping spike found in its mirror).
# ----------------------------------------------------------------------
def _cfg_depth(**kw):
    base = dict(use_growable_chunk_depth=True, chunk_deliberation_horizon=60,
                min_repetitions=3)
    base.update(kw)
    return _cfg(**base)


def _seed_depth_hierarchy(pc, whole_outcomes, sub_outcomes):
    """Register a nesting chain up to the depth ceiling and seed its tally.

    Chain is a suffix chain -- (3,4) < (2,3,4) < (1,2,3,4) ... -- because that
    is the only shape note_outcome can actually produce (it credits suffixes).
    """
    ceiling = pc.effective_max_depth
    keys = [tuple(range(k, 5)) for k in range(4 - ceiling, 4)][::-1]
    keys = sorted(keys, key=len)
    for depth, key in enumerate(keys, start=1):
        chunk = pc.accumulator.mint(key, value_tag=1.0, depth=depth)
        pc.library.register(chunk)
        pc.accumulator._tally[key] = list(
            whole_outcomes if depth == ceiling else sub_outcomes
        )
    return keys


def test_c12_depth_is_off_by_default_and_pinned_when_off():
    pc = PolicyChunking(_cfg())
    assert pc.config.use_growable_chunk_depth is False
    assert pc.effective_max_depth == pc.config.max_depth == 3
    _seed_depth_hierarchy(pc, [1.0] * 8, [0.0] * 8)
    assert pc.consider_depth_growth() is False
    assert pc.effective_max_depth == 3


def test_c12_derivation_reproduces_the_inherited_cap_at_reeds_real_horizon():
    """THE ANCHOR. A built agent must still derive exactly 3.

    Regression guard against silently raising the constant. Solway 2014 caps
    its own hierarchies at one level for stated tractability reasons and
    licenses no replacement depth, so an agent at today's deliberation budget
    must be left exactly where it was. Reads the real from_dims horizon (30),
    NOT the HippocampalConfig dataclass default (10) no built agent uses.
    """
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4,
                              use_policy_chunking=True, use_growable_chunk_depth=True)
    agent = REEAgent(cfg)
    pcfg = agent.policy_chunking.config
    assert pcfg.chunk_deliberation_horizon == cfg.hippocampal.horizon == 30
    assert pcfg.derived_chunk_max_depth == 3
    assert agent.policy_chunking.effective_max_depth == 3


def test_c12_depth_and_size_read_the_same_deliberation_budget():
    """Both parameters are settings on ONE trade-off, so one budget moves both.

    This is the property the coupled-parameter experiment depends on: raising
    the deliberation horizon must be a single independent variable, not two.
    """
    cfg = PolicyChunkingConfig(chunk_deliberation_horizon=30)
    assert (cfg.derived_chunk_ceiling, cfg.derived_chunk_max_depth) == (5, 3)
    bigger = PolicyChunkingConfig(chunk_deliberation_horizon=90)
    assert bigger.derived_chunk_ceiling > 5 and bigger.derived_chunk_max_depth > 3


@pytest.mark.parametrize("horizon,expected", [(10, 3), (29, 3), (30, 3), (40, 4),
                                              (60, 6), (300, 6)])
def test_c12_depth_scales_with_the_deliberation_budget(horizon, expected):
    """Below the anchor the inherited cap holds; above it, and only above it,
    a deeper hierarchy is derived. Saturates at chunk_depth_hard_max."""
    assert PolicyChunkingConfig(
        chunk_deliberation_horizon=horizon).derived_chunk_max_depth == expected


def test_c12_the_inherited_cap_is_actually_binding_today():
    """max_depth=3 has a real degree of freedom -- it refuses a depth-4 chunk.

    Not a formality. The MECH-321 scoping spike found its own depth_cap had
    NEVER exhibited a degree of freedom, so the literature argument about it was
    conducted over a parameter with no observed variation. This pins that
    ARC-071's cap is not in that position: at the default 2-5 size budget the
    substrate would mint depth 4, and only the cap stops it.
    """
    def formed_depths(max_depth):
        pc = PolicyChunking(_cfg(min_repetitions=5, max_depth=max_depth))
        for ep in range(300):
            seq, out = ([6, 7, 8, 1, 2], 1.0) if ep % 2 == 0 else ([3, 3, 3, 3, 3], 0.0)
            for a in seq:
                pc.record_step(a)
            pc.note_outcome(out)
            pc.end_episode()
        return sorted(c.depth for c in pc.library.all_chunks())

    assert formed_depths(3) == [1, 2, 3]
    assert formed_depths(4) == [1, 2, 3, 4]
    # ...and 5 is INERT, which is exactly the structural bound at a 2-5 budget.
    assert formed_depths(5) == [1, 2, 3, 4]


def test_c12_structural_bound_is_set_by_the_size_ceiling():
    """A depth-D chain needs D distinct sequence lengths, so depth cannot
    outrun the size budget that has to carry it."""
    pc = PolicyChunking(_cfg())
    assert pc.structural_max_depth == 4  # 5 - 2 + 1
    assert PolicyChunking(_cfg(max_chunk_size=3)).structural_max_depth == 2
    assert PolicyChunking(_cfg(min_chunk_size=3, max_chunk_size=7)).structural_max_depth == 5


def test_c12_structural_bound_rises_with_a_grown_size_ceiling():
    """THE COUPLING, mechanical rather than by analogy.

    The bound reads the accumulator's LIVE ceiling, so a size ceiling that has
    grown licenses a deeper hierarchy in the same run.
    """
    pc = PolicyChunking(_cfg(use_growable_chunk_ceiling=True,
                             chunk_deliberation_horizon=60))
    assert pc.structural_max_depth == 4
    pc.accumulator._ceiling = 8
    assert pc.structural_max_depth == 7


def test_c12_growth_requires_a_realised_marginal_return():
    pc = PolicyChunking(_cfg_depth())
    _seed_depth_hierarchy(pc, [1.0] * 8, [0.5] * 8)
    assert pc.marginal_return_at_depth_ceiling() == pytest.approx(0.5)
    assert pc.consider_depth_growth() is True
    assert pc.effective_max_depth == 4


def test_c12_no_growth_when_the_shallower_chunk_already_predicts():
    pc = PolicyChunking(_cfg_depth())
    _seed_depth_hierarchy(pc, [1.0] * 8, [1.0] * 8)
    assert pc.marginal_return_at_depth_ceiling() == pytest.approx(0.0)
    assert pc.consider_depth_growth() is False
    assert pc.effective_max_depth == 3


def test_c12_no_evidence_is_not_a_gain_of_zero():
    """gain=None must refuse even at a zero threshold -- an unattested ceiling
    would otherwise clear a >= 0.0 bar and deepen on no evidence at all."""
    pc = PolicyChunking(_cfg_depth(chunk_depth_returns_threshold=0.0))
    assert pc.marginal_return_at_depth_ceiling() is None
    assert pc.consider_depth_growth() is False


def test_c12_growth_never_exceeds_the_structural_bound():
    """The refusal that stops the ceiling being raised into inertness.

    Budget headroom to 6 is available and unused: with the size ceiling at 5
    nothing deeper than 4 could ever be minted, so growth stops at 4.
    """
    pc = PolicyChunking(_cfg_depth(chunk_depth_returns_threshold=0.0))
    assert pc.config.derived_chunk_max_depth == 6
    for _ in range(20):
        _seed_depth_hierarchy(pc, [1.0] * 8, [0.0] * 8)
        pc.consider_depth_growth()
    assert pc.effective_max_depth == pc.structural_max_depth == 4


def test_c12_growth_never_exceeds_the_derived_budget_bound():
    pc = PolicyChunking(_cfg_depth(chunk_depth_returns_threshold=0.0,
                                   max_chunk_size=12, chunk_depth_hard_max=6))
    assert pc.structural_max_depth == 11 and pc.config.derived_chunk_max_depth == 6
    for _ in range(20):
        _seed_depth_hierarchy(pc, [1.0] * 8, [0.0] * 8)
        pc.consider_depth_growth()
    assert pc.effective_max_depth == 6


def test_c12_brake_plateaus_below_both_bounds():
    """Guardrail. Growth must stop when returns flatten, NOT at a cap."""
    pc = PolicyChunking(_cfg_depth(max_chunk_size=12))
    assert pc.config.derived_chunk_max_depth == 6 and pc.structural_max_depth == 11
    for _ in range(20):
        _seed_depth_hierarchy(pc, [1.0] * 8, [1.0] * 8)
        if not pc.consider_depth_growth():
            break
    assert pc.effective_max_depth == 3
    assert pc._n_depth_growths == 0


def test_c12_growth_is_decoupled_from_the_repetition_tally():
    """Bo 2009: depth budget must not become a practice counter either.

    Unbounded repetition with zero marginal return must not deepen the ceiling.
    """
    pc = PolicyChunking(_cfg_depth())
    _seed_depth_hierarchy(pc, [1.0] * 500, [1.0] * 500)
    assert pc.consider_depth_growth() is False
    assert pc.effective_max_depth == 3


def test_c12_returns_read_the_live_tally_not_the_frozen_value_tag():
    """value_tag is frozen at formation, so a chunk whose returns have since
    collapsed would keep licensing growth on a number that stopped being true.
    """
    pc = PolicyChunking(_cfg_depth())
    keys = _seed_depth_hierarchy(pc, [1.0] * 8, [0.0] * 8)
    assert pc.marginal_return_at_depth_ceiling() == pytest.approx(1.0)
    # value_tags are all 1.0; collapse the realised outcomes at the ceiling only.
    pc.accumulator._tally[keys[-1]] = [0.0] * 8
    assert all(c.value_tag == 1.0 for c in pc.library.all_chunks())
    assert pc.marginal_return_at_depth_ceiling() == pytest.approx(0.0)


def test_c12_depth_grows_end_to_end_so_the_flag_is_not_inert():
    """Nested suffix chain where each further level genuinely pays."""
    branches = [([0, 1, 2, 3, 4], 1.0), ([9, 1, 2, 3, 4], 0.75),
                ([9, 9, 2, 3, 4], 0.5), ([9, 9, 9, 3, 4], 0.25),
                ([9, 9, 9, 9, 4], 0.0)]

    def drive(on):
        pc = PolicyChunking(_cfg(min_repetitions=5, use_growable_chunk_depth=on,
                                 chunk_deliberation_horizon=60))
        for ep in range(400):
            seq, out = branches[ep % 5]
            for a in seq:
                pc.record_step(a)
            pc.note_outcome(out)
            pc.end_episode()
        return pc

    off, on = drive(False), drive(True)
    assert off.effective_max_depth == 3 and off._n_depth_growths == 0
    assert max(c.depth for c in off.library.all_chunks()) == 3
    assert on.effective_max_depth == 4 and on._n_depth_growths >= 1
    assert max(c.depth for c in on.library.all_chunks()) == 4
    # ...and stops at the structural bound with budget headroom to 6 unused.
    assert on.config.derived_chunk_max_depth == 6
    assert on.effective_max_depth == on.structural_max_depth


def test_c12_reset_returns_the_depth_ceiling_to_the_initial_budget():
    pc = PolicyChunking(_cfg_depth())
    _seed_depth_hierarchy(pc, [1.0] * 8, [0.0] * 8)
    assert pc.consider_depth_growth() is True
    pc.reset()
    assert pc.effective_max_depth == pc.config.max_depth
    assert pc._n_depth_growths == 0


def test_c12_get_state_reports_both_bounds_separately():
    """WHICH bound binds is the substantive observation, so both are emitted."""
    state = PolicyChunking(_cfg_depth()).get_state()
    assert state["chunk_effective_max_depth"] == 3
    assert state["chunk_depth_derived_max"] == 6
    assert state["chunk_depth_structural_max"] == 4
    assert state["chunk_n_depth_growths"] == 0


@pytest.mark.parametrize("kw", [
    dict(chunk_depth_budget_fraction=0.0),
    dict(chunk_depth_budget_fraction=1.5),
    dict(chunk_depth_returns_threshold=-0.1),
    dict(chunk_depth_hard_max=2),
])
def test_c12_incoherent_depth_config_is_refused(kw):
    with pytest.raises(ValueError):
        PolicyChunkingConfig(**kw).validate()


def test_c12_from_dims_forwards_every_depth_knob():
    """All FOUR wiring sites (dataclass, signature, body, agent mapping).

    from_dims silently swallows unknown kwargs, so a knob wired at only three
    sites is unreachable with NO error -- this asserts the round-trip rather
    than the signature.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4,
        use_policy_chunking=True, use_growable_chunk_depth=True,
        chunk_deliberation_horizon=40, chunk_depth_budget_fraction=0.25,
        chunk_depth_returns_threshold=0.2, chunk_depth_hard_max=9)
    assert cfg.use_growable_chunk_depth is True
    assert cfg.chunk_depth_budget_fraction == 0.25
    assert cfg.chunk_depth_returns_threshold == 0.2
    assert cfg.chunk_depth_hard_max == 9
    pcfg = REEAgent(cfg).policy_chunking.config
    assert pcfg.use_growable_chunk_depth is True
    assert pcfg.chunk_depth_budget_fraction == 0.25
    assert pcfg.chunk_depth_returns_threshold == 0.2
    assert pcfg.chunk_depth_hard_max == 9
    assert pcfg.derived_chunk_max_depth == 9  # floor(40 * 0.25), under hard_max


# ----------------------------------------------------------------------
# C13 -- MECH-323 credit rule (V3-EXQ-810 readiness FAIL)
#
# The as-built rule credited ONLY sub-sequences ENDING at the current
# position, which starved the tally: on the exact 810 readiness cells the
# accumulator formed 7 chunks on seed 101 and ZERO on 202/303. These pin the
# flagged all-position rule AND the two properties that stop it becoming a
# spurious pass.
# ----------------------------------------------------------------------
def _trailing_only_reference(seqs, sizes=range(2, 6)):
    """What the pre-change rule tallies. The OFF path must equal this exactly."""
    ref = {}
    for i, s in enumerate(seqs):
        outcome = 1.0 if i % 2 == 0 else 0.0
        for size in sizes:
            if len(s) < size:
                break
            ref.setdefault(tuple(s[-size:]), []).append(outcome)
    return ref


def _drive_seqs(pc, seqs):
    for i, s in enumerate(seqs):
        for a in s:
            pc.record_step(a)
        pc.note_outcome(1.0 if i % 2 == 0 else 0.0)
        pc.end_episode()


def test_c13_credit_rule_defaults_off_and_round_trips_to_the_operator():
    """FOUR wiring sites. from_dims swallows unknown kwargs silently."""
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_chunk_all_position_credit is False
    assert PolicyChunkingConfig().use_chunk_all_position_credit is False

    on = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=4,
        use_policy_chunking=True, use_chunk_all_position_credit=True)
    assert on.use_chunk_all_position_credit is True
    assert REEAgent(on).policy_chunking.config.use_chunk_all_position_credit is True


def test_c13_off_path_is_bit_identical_to_trailing_only():
    """The OFF path must reproduce the pre-change tally EXACTLY, not merely
    'behave similarly' -- this is what makes the flag safe to land."""
    seqs = [(0, 1, 2, 3) if i % 2 == 0 else (3, 0, 3, 0) for i in range(60)]
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60,
                             use_chunk_all_position_credit=False))
    _drive_seqs(pc, seqs)
    assert {k: list(v) for k, v in pc.accumulator._tally.items()} == \
        _trailing_only_reference(seqs)


def test_c13_on_credits_non_trailing_subsequences():
    """The starvation being fixed: a LEADING pair is never tallied when off."""
    seqs = [(0, 1, 2, 3)] * 10
    off = PolicyChunking(_cfg(min_repetitions=5, use_chunk_all_position_credit=False))
    _drive_seqs(off, seqs)
    on = PolicyChunking(_cfg(min_repetitions=5, use_chunk_all_position_credit=True))
    _drive_seqs(on, seqs)
    assert (0, 1) not in off.accumulator._tally
    assert (0, 1) in on.accumulator._tally
    assert len(on.accumulator._tally) > len(off.accumulator._tally)


def test_c13_a_key_takes_at_most_one_credit_per_outcome():
    """min_repetitions counts repetition ACROSS trials, never within one.

    Without this a single 5-long held run would advance the tally by 4 and
    drive variance to 0 -- manufacturing the (reps >= R_min AND var < F_low)
    conjunction the formation gate exists to test.
    """
    pc = PolicyChunking(_cfg(min_repetitions=5, use_chunk_all_position_credit=True))
    for a in (1, 1, 1, 1, 1):
        pc.record_step(a)
    pc.note_outcome(1.0)
    assert len(pc.accumulator._tally[(1, 1)]) == 1
    assert len(pc.accumulator._tally[(1, 1, 1)]) == 1
    assert pc.get_state()["chunk_acc_n_formed"] == 0


def test_c13_a_single_held_run_cannot_form_a_chunk():
    """The spurious-PASS guard, stated at the level a readiness criterion reads.

    Crediting a held-action stream at every position WITHOUT the per-outcome
    dedup mints run-chunks ((1,1,1,1,1), variance identically 0) that would
    satisfy C1 for an entirely spurious reason.
    """
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60,
                             use_chunk_all_position_credit=True))
    for _ in range(3):
        for a in (1, 1, 1, 1, 1):
            pc.record_step(a)
        pc.note_outcome(1.0)
        pc.end_episode()
    # 3 trials of a 5-run: 3 credits per key, still short of R_min = 5.
    assert len(pc.accumulator._tally[(1, 1)]) == 3
    assert pc.get_state()["chunk_acc_n_formed"] == 0


def test_c13_was_executed_moves_with_the_credit_rule():
    """Coupled by design: if corroboration stayed trailing-only, chunks formed
    at non-trailing positions would form and then never crystallise -- a C1
    failure converted into a C2 one."""
    chunk = ChunkedPrimitive(sequence=(0, 1), initiation_set=(), depth=1)
    off = PolicyChunking(_cfg(use_chunk_all_position_credit=False))
    on = PolicyChunking(_cfg(use_chunk_all_position_credit=True))
    for pc in (off, on):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
    assert off._was_executed(chunk) is False   # (0,1) does not END the buffer
    assert on._was_executed(chunk) is True     # but it IS in the buffer
    # the trailing sequence is corroborated under BOTH rules
    trailing = ChunkedPrimitive(sequence=(2, 3), initiation_set=(), depth=1)
    assert off._was_executed(trailing) is True
    assert on._was_executed(trailing) is True


def test_c13_uniform_outcomes_still_form_nothing_under_the_new_rule():
    """C6's relativity contract must survive the credit-rule change."""
    pc = PolicyChunking(_cfg(min_repetitions=5, window_trials=60,
                             use_chunk_all_position_credit=True))
    for _ in range(60):
        for a in (0, 1, 2, 3):
            pc.record_step(a)
        pc.note_outcome(1.0)
        pc.end_episode()
    assert pc.get_state()["chunk_acc_n_formed"] == 0
