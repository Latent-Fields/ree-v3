"""ARC-070 policy_decomposition_via_event_segmenter (MECH-321) -- interface contracts.

Interface-level guarantees, NOT magnitude thresholds. The magnitudes (how
often decomposition fires, whether execution-time prediction failures drop)
belong to the queued validation experiment, not here.

C1  config defaults + from_dims surfaces them + master OFF is bit-identical
    (no hippocampal sub-config mirror needed -- see config.py field docstring)
C2  loud precondition: use_policy_decomposition=True requires
    hippocampal.use_event_segmenter=True
C3  decompose_sequence(): a depth-1 sequence unpacks to raw-action leaves
    (regression pin -- this session's own activation smoke test caught a
    `depth <= 1` early-return bug that made depth-1 chunks, the common case,
    permanently un-decomposable; see policy_decomposition.py history)
C4  decompose_sequence(): depth>1 tiles against a registered shallower
    sub-chunk in the library (longest-match) before falling back to raw
    actions for uncovered positions
C5  decompose_sequence(): an atomic (len<=1) sequence returns no tiles
C6  evaluate(): R1 is an OR trigger -- V_s alone, boundary alone, or neither
C7  evaluate(): R3 depth_cap -- a primitive at/above depth_cap that should
    decompose is marked_unreliable, not decomposed further
C8  live agent (pre-commit): a withheld ARC-071 chunk candidate is replaced
    in the CEM pool by its leaf-tile candidates, not silently dropped
C9  live agent (mid-execution, R4 second phase): a triggering remainder
    releases the beta-gate commit latch
C10 chunk injection stays additive: an untriggered chunk candidate passes
    through the pool unchanged
"""

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy import (
    ChunkedPrimitive,
    ChunkLibrary,
    ChunkState,
    DecompositionDecision,
    PolicyChunking,
    PolicyDecomposition,
    PolicyDecompositionConfig,
)
from ree_core.utils.config import REEConfig


class _FakeBoundary:
    def __init__(self, fired: bool, posterior: float = 0.0):
        self.fired = fired
        self.posterior = posterior
        self.events = []


class _FakeSegmenter:
    """Stub MECH-288 -- returns a scripted boundary_on() result."""

    def __init__(self, fired: bool = False, posterior: float = 0.0):
        self._fired = fired
        self._posterior = posterior
        self.calls = []

    def boundary_on(self, stream, latent, pe=None, t=None):
        self.calls.append({"stream": stream, "latent": latent, "pe": pe, "t": t})
        return _FakeBoundary(self._fired, self._posterior)


class _FakeDecompositionSource:
    """Stub PolicyDecomposition -- returns a scripted evaluate() result
    every call, for deterministic (non-stochastic-rollout-dependent) tests
    of HippocampalModule's pool-filtering wiring."""

    def __init__(self, should_decompose: bool):
        self._should_decompose = should_decompose
        self.n_calls = 0

    def evaluate(self, **kwargs):
        self.n_calls += 1
        return DecompositionDecision(
            should_decompose=self._should_decompose,
            decomposed=False,
            marked_unreliable=False,
            v_s=1.0,
            boundary_fired=False,
            boundary_posterior=0.0,
            depth=int(kwargs.get("depth", 1)),
            hypothesis_tag=bool(kwargs.get("hypothesis_tag", True)),
        )


def _env_agent(**cfg_kwargs):
    env = CausalGridWorldV2()
    _, obs_dict = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        reafference_action_dim=env.action_dim,
        **cfg_kwargs,
    )
    agent = REEAgent(cfg)
    return env, obs_dict, agent


# ----------------------------------------------------------------------
# C1 -- config surface + OFF is bit-identical
# ----------------------------------------------------------------------
def test_c1_defaults_are_off_and_from_dims_forwards():
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=4)
    assert cfg.use_policy_decomposition is False
    assert cfg.decomposition_vs_threshold == 0.4
    assert cfg.decomposition_depth_cap == 3

    agent = REEAgent(cfg)
    assert agent.policy_decomposition is None
    assert agent.get_policy_decomposition_state() == {}


def test_c1_from_dims_forwards_non_default_values():
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_event_segmenter=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.7,
        decomposition_depth_cap=4,
    )
    assert cfg.use_policy_decomposition is True
    assert cfg.decomposition_vs_threshold == 0.7
    assert cfg.decomposition_depth_cap == 4
    agent = REEAgent(cfg)
    assert agent.policy_decomposition is not None
    assert agent.policy_decomposition.config.vs_decompose_threshold == 0.7
    assert agent.policy_decomposition.config.depth_cap == 4


# ----------------------------------------------------------------------
# C2 -- loud precondition
# ----------------------------------------------------------------------
def test_c2_requires_event_segmenter():
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=16,
        action_dim=4,
        use_policy_decomposition=True,  # use_event_segmenter left OFF
    )
    try:
        REEAgent(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "use_event_segmenter" in str(exc)


# ----------------------------------------------------------------------
# C3 -- depth-1 decomposes to raw actions (regression pin)
# ----------------------------------------------------------------------
def test_c3_depth1_sequence_unpacks_to_raw_actions():
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=1, library=None)
    assert tiles == (((0,), 0), ((1,), 0), ((2,), 0))


# ----------------------------------------------------------------------
# C4 -- library-aware tiling at depth > 1
# ----------------------------------------------------------------------
def test_c4_tiles_against_registered_shallower_chunk():
    library = ChunkLibrary()
    sub = ChunkedPrimitive(sequence=(0, 1), depth=1, state=ChunkState.CRYSTALLISED)
    library.register(sub)
    # depth-2 sequence = [sub-chunk (0,1)] + [raw action 2]
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=2, library=library)
    assert tiles == (((0, 1), 1), ((2,), 0))


def test_c4_dissolved_chunks_are_not_tiled_against():
    library = ChunkLibrary()
    sub = ChunkedPrimitive(sequence=(0, 1), depth=1, state=ChunkState.DISSOLVED)
    library.register(sub)
    tiles = PolicyDecomposition.decompose_sequence((0, 1, 2), depth=2, library=library)
    # No live sub-chunk covers (0, 1) -> falls back to raw actions throughout.
    assert tiles == (((0,), 0), ((1,), 0), ((2,), 0))


# ----------------------------------------------------------------------
# C5 -- atomic sequence has no tiles
# ----------------------------------------------------------------------
def test_c5_atomic_sequence_returns_no_tiles():
    assert PolicyDecomposition.decompose_sequence((0,), depth=1, library=None) == ()
    assert PolicyDecomposition.decompose_sequence((), depth=1, library=None) == ()


# ----------------------------------------------------------------------
# C6 -- R1 OR trigger
# ----------------------------------------------------------------------
def test_c6_vs_alone_triggers():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.5))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.1,  # below threshold
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is True
    assert decision.v_s == 0.1
    assert decision.boundary_fired is False


def test_c6_boundary_alone_triggers():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.1))
    seg = _FakeSegmenter(fired=True, posterior=0.9)
    decision = pd.evaluate(
        region_vs=0.9,  # comfortably above threshold
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is True
    assert decision.boundary_fired is True
    assert decision.boundary_posterior == 0.9


def test_c6_neither_triggers_nothing():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.1))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.9,
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=1,
        sequence=(0, 1),
    )
    assert decision.should_decompose is False
    assert decision.decomposed is False
    assert decision.marked_unreliable is False


# ----------------------------------------------------------------------
# C7 -- depth_cap marks unreliable instead of decomposing
# ----------------------------------------------------------------------
def test_c7_depth_at_cap_marks_unreliable():
    pd = PolicyDecomposition(PolicyDecompositionConfig(vs_decompose_threshold=0.9, depth_cap=2))
    seg = _FakeSegmenter(fired=False)
    decision = pd.evaluate(
        region_vs=0.0,  # triggers
        latent_signature={"z_world": None},
        event_segmenter=seg,
        depth=2,  # == depth_cap
        sequence=(0, 1, 2),
    )
    assert decision.should_decompose is True
    assert decision.marked_unreliable is True
    assert decision.decomposed is False
    assert decision.sub_elements == ()
    assert pd.get_state()["decomp_n_marked_unreliable"] == 1
    assert pd.get_state()["decomp_n_decomposed_precommit"] == 0


# ----------------------------------------------------------------------
# C8 -- live agent, pre-commit: withheld chunk is replaced by leaf tiles
# ----------------------------------------------------------------------
def test_c8_withheld_chunk_replaced_by_leaf_tiles_in_pool():
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,  # near-certain trigger
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("mech321_chunks_withheld", 0) >= 1
    assert diag.get("mech321_chunks_decomposed", 0) >= 1
    assert diag.get("mech321_leaf_tiles_added", 0) >= 1
    # The full chunk sequence must not survive as a single atomic candidate.
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) not in surviving
    state = agent.get_policy_decomposition_state()
    assert state["decomp_n_evaluated_precommit"] > 0
    assert state["decomp_n_decomposed_precommit"] >= 1


# ----------------------------------------------------------------------
# C9 -- live agent, mid-execution: triggering remainder releases the latch
# ----------------------------------------------------------------------
def test_c9_midexecution_trigger_releases_commit_latch():
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
        decomposition_vs_threshold=0.99,
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

    fake_traj = candidates[0]
    fake_traj.metadata = {
        "source": "arc071_chunk",
        "chunk_sequence": list(seq),
        "chunk_depth": 1,
    }
    agent.e3._committed_trajectory = fake_traj
    agent._committed_step_idx = 1  # 2 steps remaining -- above the >1 floor
    agent.beta_gate.elevate()
    assert agent.beta_gate.is_elevated

    agent.select_action(candidates, ticks)

    assert agent.beta_gate.is_elevated is False
    assert agent.get_policy_decomposition_state()["decomp_n_evaluated_midexec"] >= 1


# ----------------------------------------------------------------------
# C10 -- injection stays additive when nothing triggers
# ----------------------------------------------------------------------
def test_c10_untriggered_chunk_passes_through_unchanged():
    """Deterministic version of the passthrough guarantee: stub the
    decomposition source to always report should_decompose=False (rather
    than relying on a real, stochastically-rolled-out E2 predictor to stay
    quiet over an 8-tick sweep -- see C8/C9 for the live-segmenter path)."""
    env, obs_dict, agent = _env_agent(
        use_event_segmenter=True,
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        use_policy_decomposition=True,
    )
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    fake_source = _FakeDecompositionSource(should_decompose=False)
    agent.hippocampal.set_decomposition_source(fake_source)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    assert fake_source.n_calls > 0
    diag = agent.hippocampal._last_propose_diagnostics
    assert diag.get("arc071_chunk_candidates_added", 0) >= 1
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) in surviving
    assert diag.get("mech321_chunks_withheld", 0) == 0


def test_c10_no_decomposition_source_is_bit_identical():
    """set_decomposition_source() never called -> _apply_policy_decomposition
    is a no-op passthrough, matching pre-MECH-321 ARC-071-only behaviour."""
    env, obs_dict, agent = _env_agent(
        use_policy_chunking=True,
        use_chunk_proposal_injection=True,
        # use_policy_decomposition left OFF: agent.policy_decomposition is
        # None, set_decomposition_source() is never called.
    )
    assert agent.policy_decomposition is None
    seq = (0, 1, 2)
    chunk = ChunkedPrimitive(
        sequence=seq, depth=1, state=ChunkState.CRYSTALLISED, selection_weight=1.0
    )
    agent.policy_chunking.library.register(chunk)

    obs_body, obs_world = obs_dict["body_state"], obs_dict["world_state"]
    latent = agent.sense(obs_body, obs_world)
    ticks = agent.clock.advance()
    e1_prior = (
        agent._e1_tick(latent)
        if ticks["e1_tick"]
        else torch.zeros(1, 32, device=agent.device)
    )
    agent.generate_trajectories(latent, e1_prior, ticks)

    diag = agent.hippocampal._last_propose_diagnostics
    surviving = diag.get("arc071_chunk_sequences", [])
    assert list(seq) in surviving
    assert "mech321_chunks_withheld" not in diag
